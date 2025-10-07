import asyncio, os, re, json, shutil
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
from dotenv import load_dotenv; load_dotenv()

from azure.identity.aio import AzureCliCredential
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters, types as mcp_types

# ---------------- env helpers ----------------
def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env: {name}")
    return v

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KNOW = os.path.join(BASE_DIR, "knowledge")
KNOW_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge")  # keep VERBATIM (match local mcp which works)
ABS_KNOW_DIR = os.path.abspath(os.path.join(BASE_DIR, KNOW_DIR))
KNOW_BASENAME = os.path.basename(ABS_KNOW_DIR.rstrip("\\/")) or "knowledge"

DEBUG = os.getenv("DEBUG_DOCS") == "1"
ALLOW_FS_FALLBACK = os.getenv("ALLOW_FS_FALLBACK") == "1"

def _clip(s: str, n=4000) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "..."

def _urls(t: str) -> List[str]:
    return re.findall(r"https?://[^\s)]+", t or "")

def _log(*a):
    if DEBUG: print("[docs]", *a)

def _resolve_npx_cmd() -> str:
    if os.getenv("MCP_FS_CMD"):
        return os.getenv("MCP_FS_CMD")
    p = shutil.which("npx")
    if p: return p
    if os.name == "nt":
        where_node = shutil.which("node")
        if where_node:
            cand = os.path.join(os.path.dirname(where_node), "npx.cmd")
            if os.path.isfile(cand):
                return cand
    # last resort
    return "npx"

# ---------------- MCP stdio ----------------
@dataclass
class MCP:
    cmd: str
    args: List[str]  # keep as list (for Windows esp)

@asynccontextmanager
async def mcp_connect(spec: MCP):
    if DEBUG:
        print(f"[mcp] launching: {spec.cmd} {' '.join(spec.args)}")
    async with stdio_client(StdioServerParameters(command=spec.cmd, args=spec.args)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            yield s

async def call_tool(s: ClientSession, names: List[str], args: Dict[str, Any]) -> Optional[mcp_types.CallToolResult]:
    tools = await s.list_tools()
    available = {t.name for t in tools.tools}
    if DEBUG: print("MCP tools:", sorted(available))
    for n in names:
        if n in available:
            _log("call", n, "args=", args)
            return await s.call_tool(n, arguments=args)
    _log("no matching tool found in", names)
    return None

def to_text(res: Optional[mcp_types.CallToolResult]) -> str:
    if not res: return ""
    if getattr(res, "structuredContent", None):
        try: return json.dumps(res.structuredContent, indent=2)
        except Exception: pass
    for c in res.content:
        if isinstance(c, mcp_types.TextContent): return c.text
    return ""

# ---------------- MCP server configs ----------------
PUBMED = MCP(
    os.getenv("MCP_PUBMED_CMD", _resolve_npx_cmd()),
    os.getenv("MCP_PUBMED_ARGS", "@cyanheads/pubmed-mcp-server --stdio").split()
)
CTGOV = MCP(
    os.getenv("MCP_CT_CMD", _resolve_npx_cmd()),
    os.getenv("MCP_CT_ARGS", "clinicaltrialsgov-mcp-server --stdio").split()
)
FETCH = MCP(
    os.getenv("MCP_FETCH_CMD", _resolve_npx_cmd()),
    os.getenv("MCP_FETCH_ARGS", "@sylphlab/tools-fetch-mcp --stdio").split()
)

# Filesystem: positional root ONLY (exactly like the smoke test)
if os.getenv("MCP_FS_ARGS"):
    FS_ARGS = os.getenv("MCP_FS_ARGS").split()
else:
    # Use the ENV value verbatim (e.g., "knowledge") to match your working command.
    FS_ARGS = ["@modelcontextprotocol/server-filesystem", KNOW_DIR]
FILES = MCP(os.getenv("MCP_FS_CMD", _resolve_npx_cmd()), FS_ARGS)

# ---------------- prompts ----------------
ORCHESTRATOR_SYS = """You are the Orchestrator. Merge agent outputs into a concise answer.
Order sections: Evidence, Trials, Guidelines, Safety, Local Policy (if any policy available). Less than 10 sentences in total.
Deduplicate facts and citations when needed. End with one 'Sources' line listing unique PMIDs, NCT IDs, and URLs. 
Finally add a brief suumary addressing the question summarizing Evidence, Trials, Guidelines, Safety, Local Policy and fiving references.
"""
LITERATURE_SYS = """You are the Literature Agent for PubMed. You will be given raw PubMed snippets.
Return 3 bullets of key efficacy findings with PMIDs like (PMID:xxxxxxx). Be concise. Do not invent PMIDs.
"""
TRIALS_SYS = """You are the Trials Agent for ClinicalTrials.gov. You will be given raw study hits.
Return up to 3 notable completed or phase 3 trials with NCT IDs and a 1-line outcome. Be concise.
"""
GUIDELINES_SYS = """You are the Guidelines Agent. You will get raw guideline text fetched from the web.
Extract exactly 2 actionable points and include the URLs. Prefer ADA/AACE/NICE/WHO quality sources. Be concise.
"""
SAFETY_SYS = """You are the Safety Agent. You will be given raw PubMed safety snippets.
List 2–3 key adverse events and warnings with 1 short monitoring tip. Cite PMIDs if present. Be concise.
"""
DOCS_SYS = """You are the Document Agent. You will receive local policy text blobs.
Restate 1–2 relevant local rules and include the filenames. If none apply, say 'no local policy'.
"""

# ---------------- agent factory ----------------
async def mk_agent(name: str, sys_prompt: str) -> ChatAgent:
    cred = AzureCliCredential()
    client = AzureAIAgentClient(async_credential=cred)  # uses AZURE_AI_* env. Add as needed.
    a = ChatAgent(chat_client=client, instructions=sys_prompt, name=name) # new framework. multiturn
    a._cred = cred
    return a

# ---------------- role tasks ----------------
async def literature_task(q: str) -> str:
    query = f"{q} randomized controlled trial OR RCT OR meta-analysis"
    hits = details = ""
    try:
        async with mcp_connect(PUBMED) as s:
            hits  = to_text(await call_tool(s, ["pubmed_search_articles","pubmed.search"], {"query": query, "retmax": 5, "sort": "relevance"}))
            pmids = set(re.findall(r"\bPMID[:\s]*([0-9]{5,9})\b", hits))
            if pmids:
                details = to_text(await call_tool(s, ["pubmed_fetch_contents","pubmed.fetch"], {"pmids": list(pmids)[:3], "detailLevel":"abstract_plus"}))
    except Exception as e:
        hits = f"(pubmed error: {e})"
    A = await mk_agent("LiteratureAgent", LITERATURE_SYS)
    async with A, A._cred:
        return await A.run(f"User query: {q}\n\nSearch hits:\n{_clip(hits)}\n\nDetails:\n{_clip(details)}")

async def safety_task(q: str) -> str:
    try:
        async with mcp_connect(PUBMED) as s:
            hits = to_text(await call_tool(s, ["pubmed_search_articles","pubmed.search"], {"query": f"{q} adverse events OR safety OR pancreatitis OR gallbladder", "retmax": 5, "sort":"relevance"}))
    except Exception as e:
        hits = f"(pubmed safety error: {e})"
    A = await mk_agent("SafetyAgent", SAFETY_SYS)
    async with A, A._cred:
        return await A.run(f"User query: {q}\n\nSafety hits:\n{_clip(hits)}")

async def trials_task(q: str) -> str:
    try:
        async with mcp_connect(CTGOV) as s:
            hits = to_text(await call_tool(s, ["clinicaltrials_list_studies","clinicaltrials_search_studies","ctgov.search"], {"query": f"{q} phase 3 OR randomized | condition: obesity | intervention: GLP-1", "max_results": 5}))
    except Exception as e:
        hits = f"(ctgov error: {e})"
    A = await mk_agent("TrialsAgent", TRIALS_SYS)
    async with A, A._cred:
        return await A.run(f"User query: {q}\n\nTrials hits:\n{_clip(hits)}")

async def guidelines_task(q: str) -> str:
    planner = await mk_agent("GuidelinePlanner", "You pick 1–2 authoritative guideline URLs (ADA/AACE/NICE/WHO) for the topic. Return ONLY raw URLs separated by spaces.")
    async with planner, planner._cred:
        urls_text = str(await planner.run(f"Topic: {q}"))
    urls = _urls(urls_text)[:2]
    blobs: List[str] = []
    try:
        async with mcp_connect(FETCH) as s:
            for u in urls:
                body = to_text(await call_tool(s, ["fetch_markdown","fetch_txt","fetch_text","fetch_html"], {"url": u}))
                blobs.append(f"URL: {u}\n{_clip(body, 5000)}")
    except Exception as e:
        blobs.append(f"(fetch error: {e})")
    A = await mk_agent("GuidelinesAgent", GUIDELINES_SYS)
    async with A, A._cred:
        return await A.run(f"User query: {q}\n\n" + ("\n\n".join(blobs) if blobs else "No URLs fetched."))

# ---------------- Docs task (files in root - not perfect) ----------------
async def docs_task(q: str) -> str:
    print(f"DEBUG: Checking knowledge dir: {ABS_KNOW_DIR}")
    print(f"DEBUG: Directory exists: {os.path.isdir(ABS_KNOW_DIR)}")
    if not os.path.isdir(ABS_KNOW_DIR):
        A = await mk_agent("DocsAgent", DOCS_SYS)
        async with A, A._cred:
            return await A.run("No local policy. The folder was not found.")

    CAND_LIST = ["list_directory", "filesystem.list_directory", "fs.list", "list", "dir", "files.list"]
    CAND_READ = ["read_file", "filesystem.read_file", "fs.read", "read", "file.read", "files.read"]

    picks: List[str] = []

    try:
        print("DEBUG: Attempting to connect to MCP server...")
        async with mcp_connect(FILES) as s:
            print("DEBUG: Connected to MCP server successfully")

            # --- 1) LIST root (relative-only) and parse both list and object forms ---
            items: List[str] = []
            for pv in ["", "."]:
                res = await call_tool(s, CAND_LIST, {"path": pv, "recursive": False})
                listing_text = to_text(res) if res else ""
                _log("listing raw:", listing_text[:200].replace("\n", " ⏎ "))

                if not listing_text:
                    continue
                if "Access" in listing_text and "denied" in listing_text:
                    _log(f"skip error listing for path {pv!r}: {listing_text[:120]}")
                    continue

                parsed_any = False
                try:
                    j = json.loads(listing_text)
                    # Case A: top-level array
                    if isinstance(j, list):
                        for it in j:
                            if isinstance(it, dict):
                                name = (it.get("name") or it.get("path") or "").strip()
                                is_dir = bool(it.get("isDir")) if isinstance(it, dict) else False
                                if name and not is_dir:
                                    items.append(name)
                            elif isinstance(it, str):
                                items.append(it.strip())
                        parsed_any = True
                    # Case B: object with entries/files/items arrays
                    elif isinstance(j, dict):
                        for key in ("entries", "files", "items"):
                            seq = j.get(key)
                            if isinstance(seq, list):
                                for it in seq:
                                    if isinstance(it, dict):
                                        name = (it.get("name") or it.get("path") or "").strip()
                                        is_dir = bool(it.get("isDir")) if isinstance(it, dict) else False
                                        if name and not is_dir:
                                            items.append(name)
                                    elif isinstance(it, str):
                                        items.append(it.strip())
                                parsed_any = True
                                break
                except Exception:
                    pass

                # Fallback: token scan (keep only .txt/.md)
                if not parsed_any:
                    tokens = re.findall(r"[^\s]+", listing_text)
                    items.extend([t for t in tokens if t.lower().endswith((".txt", ".md"))])

                if items:
                    break  # got something

            print(f"DEBUG: Found raw items: {items}")

            # Normalize to clean basenames at root
            norm = []
            for it in items:
                base = os.path.basename(it.strip())
                if base and base.lower().endswith(('.txt', '.md')) and not any(x in base for x in ['..', ':', '\\', '/']):
                    norm.append(base)
            print(f"DEBUG: Normalized items: {norm}")

            # If still empty, discover locally (names only); we'll still read via MCP
            if not norm:
                for _root, _dirs, files in os.walk(ABS_KNOW_DIR):
                    for fn in files:
                        if fn.lower().endswith((".txt", ".md")):
                            norm.append(os.path.basename(fn))

            # Prioritize discharge* then others (max 3)
            discharge_first = [f for f in norm if "discharge" in f.lower()]
            others = [f for f in norm if f not in discharge_first]
            ordered = discharge_first + others

            seen = set()
            for f in ordered:
                if f not in seen:
                    seen.add(f); picks.append(f)
                if len(picks) >= 3:
                    break

            if not picks:
                A = await mk_agent("DocsAgent", DOCS_SYS)
                async with A, A._cred:
                    return await A.run("no local policy")

            #2) READ with basenames only 
            blobs = []
            for name in picks:
                content = ""
                res = await call_tool(s, CAND_READ, {"path": name})
                if res:
                    content = to_text(res) or ""

                if not content and ALLOW_FS_FALLBACK:
                    local_path = os.path.join(ABS_KNOW_DIR, name)
                    if os.path.isfile(local_path):
                        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                blobs.append(f"{name}:\n{_clip(content or '(empty)', 2000)}")

    except Exception as e:
        blobs = [f"(filesystem error: {e})"]

    A = await mk_agent("DocsAgent", DOCS_SYS)
    async with A, A._cred:
        return await A.run(f"User question: {q}\n\n" + "\n\n".join(blobs))

# ---------------- Orchestrator ----------------
async def orchestrate(q: str, include_docs: bool = True) -> str:
    lit, trl, gdl, sft, doc = await asyncio.gather(
        literature_task(q), trials_task(q), guidelines_task(q), safety_task(q),
        docs_task(q) if include_docs else asyncio.sleep(0, result="no local policy")
    )
    merged = (
        f"User question: {q}\n\n"
        f"=== Evidence ===\n{lit}\n\n=== Trials ===\n{trl}\n\n"
        f"=== Guidelines ===\n{gdl}\n\n=== Safety ===\n{sft}\n\n=== Local Policy ===\n{doc}\n"
    )
    O = await mk_agent("Orchestrator", ORCHESTRATOR_SYS)
    async with O, O._cred:
        return await O.run(merged)

# ---------------- CLI ----------------
async def main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: python clinical_workflow_mcp.py "what is the discharge hospital policy"')
        raise SystemExit(2)
    _req("AZURE_AI_PROJECT_ENDPOINT"); _req("AZURE_AI_MODEL_DEPLOYMENT_NAME")
    print(f"[info] KNOWLEDGE_DIR (verbatim): {KNOW_DIR}  -> ABS: {ABS_KNOW_DIR}")
    print(await orchestrate(sys.argv[1], include_docs=True))

if __name__ == "__main__":
    asyncio.run(main())
