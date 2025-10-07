#!/usr/bin/env python3
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

# ---------- paths & env ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.getenv("PROMPTS_DIR", os.path.join(BASE_DIR, "prompts"))
KNOW_DIR_VERBATIM = os.getenv("KNOWLEDGE_DIR", "knowledge")  # keep as provided (relative recommended)
ABS_KNOW_DIR = os.path.abspath(os.path.join(BASE_DIR, KNOW_DIR_VERBATIM))

DEBUG = os.getenv("DEBUG_DOCS") == "1"
ALLOW_FS_FALLBACK = os.getenv("ALLOW_FS_FALLBACK") == "1"

def _req(name: str) -> str:
    v = os.getenv(name); 
    if not v: raise RuntimeError(f"Missing env: {name}")
    return v

def _clip(s: str, n=4000) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "..."

def _urls(t: str) -> List[str]:
    return re.findall(r"https?://[^\s)]+", t or "")

def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ---------- prompt files ----------
ORCHESTRATOR_SYS = load_prompt("orchestrator.txt")
LITERATURE_SYS   = load_prompt("literature.txt")
TRIALS_SYS       = load_prompt("trials.txt")
GUIDELINES_SYS   = load_prompt("guidelines.txt")
SAFETY_SYS       = load_prompt("safety.txt")
DOCS_SYS         = load_prompt("docs.txt")

# ---------- MCP plumbing ----------
@dataclass
class MCP:
    cmd: str
    args: List[str]

def _resolve_npx_cmd() -> str:
    if os.getenv("MCP_FS_CMD"):  # honor explicit override
        return os.getenv("MCP_FS_CMD")
    p = shutil.which("npx")
    if p: return p
    if os.name == "nt":
        n = shutil.which("node")
        if n:
            cand = os.path.join(os.path.dirname(n), "npx.cmd")
            if os.path.isfile(cand): return cand
    return "npx"

@asynccontextmanager
async def mcp_connect(spec: MCP):
    if DEBUG: print(f"[mcp] launch: {spec.cmd} {' '.join(map(str,spec.args))}")
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
            return await s.call_tool(n, arguments=args)
    return None

def to_text(res: Optional[mcp_types.CallToolResult]) -> str:
    if not res: return ""
    if getattr(res, "structuredContent", None):
        try: return json.dumps(res.structuredContent, indent=2)
        except Exception: pass
    for c in res.content:
        if isinstance(c, mcp_types.TextContent): return c.text
    return ""

# ---------- MCP servers ----------
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
# Filesystem server uses positional root ("knowledge") to match sandbox
FS_ARGS = os.getenv("MCP_FS_ARGS").split() if os.getenv("MCP_FS_ARGS") else [
    "@modelcontextprotocol/server-filesystem", KNOW_DIR_VERBATIM
]
FILES = MCP(os.getenv("MCP_FS_CMD", _resolve_npx_cmd()), FS_ARGS)

# ---------- Agent factory ----------
async def mk_agent(name: str, sys_prompt: str) -> ChatAgent:
    cred = AzureCliCredential()
    client = AzureAIAgentClient(async_credential=cred)
    a = ChatAgent(chat_client=client, instructions=sys_prompt, name=name)
    a._cred = cred
    return a

# ---------- Role tasks ----------
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
            items: List[str] = []
            for pv in ["", "."]:
                res = await call_tool(s, CAND_LIST, {"path": pv, "recursive": False})
                listing_text = to_text(res) if res else ""
                if not listing_text: continue
                if "Access" in listing_text and "denied" in listing_text: continue

                parsed = False
                try:
                    j = json.loads(listing_text)
                    if isinstance(j, list):
                        for it in j:
                            if isinstance(it, dict):
                                name = (it.get("name") or it.get("path") or "").strip()
                                if name: items.append(name)
                            elif isinstance(it, str):
                                items.append(it.strip())
                        parsed = True
                    elif isinstance(j, dict):
                        for key in ("entries","files","items"):
                            seq = j.get(key)
                            if isinstance(seq, list):
                                for it in seq:
                                    if isinstance(it, dict):
                                        name = (it.get("name") or it.get("path") or "").strip()
                                        if name: items.append(name)
                                    elif isinstance(it, str):
                                        items.append(it.strip())
                                parsed = True; break
                except Exception:
                    pass
                if not parsed:
                    tokens = re.findall(r"[^\s]+", listing_text)
                    items.extend([t for t in tokens if t.lower().endswith((".txt", ".md"))])
                if items: break

            norm = []
            for it in items:
                base = os.path.basename(it.strip())
                if base and base.lower().endswith(('.txt', '.md')) and not any(x in base for x in ['..', ':', '\\', '/']):
                    norm.append(base)

            if not norm:
                for _root, _dirs, files in os.walk(ABS_KNOW_DIR):
                    for fn in files:
                        if fn.lower().endswith((".txt", ".md")):
                            norm.append(os.path.basename(fn))

            discharge_first = [f for f in norm if "discharge" in f.lower()]
            others = [f for f in norm if f not in discharge_first]
            ordered = discharge_first + others

            seen = set()
            for f in ordered:
                if f not in seen:
                    seen.add(f); picks.append(f)
                if len(picks) >= 3: break

            if not picks:
                A = await mk_agent("DocsAgent", DOCS_SYS)
                async with A, A._cred:
                    return await A.run("no local policy")

            blobs = []
            for name in picks:
                content = ""
                res = await call_tool(s, CAND_READ, {"path": name})
                if res: content = to_text(res) or ""
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

# ---------- Orchestrator ----------
async def orchestrate(q: str, include_docs: bool = True) -> str:
    lit, trl, gdl, sft, doc = await asyncio.gather(
        literature_task(q), trials_task(q), guidelines_task(q), safety_task(q),
        docs_task(q) if include_docs else asyncio.sleep(0, result="no local policy")
    )
    merged = (
        f"User question: {q}\n\n"
        f"=== Literature ===\n{lit}\n\n=== Trials ===\n{trl}\n\n"
        f"=== Guidelines ===\n{gdl}\n\n=== Safety ===\n{sft}\n\n=== Local Policy ===\n{doc}\n"
    )
    O = await mk_agent("Orchestrator", ORCHESTRATOR_SYS)
    async with O, O._cred:
        return await O.run(merged)

# ---------- CLI ----------
async def main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: python main.py "your clinical question"')
        raise SystemExit(2)
    _req("AZURE_AI_PROJECT_ENDPOINT"); _req("AZURE_AI_MODEL_DEPLOYMENT_NAME")
    print(f"[info] PROMPTS_DIR: {PROMPTS_DIR}")
    print(f"[info] KNOWLEDGE_DIR (verbatim): {KNOW_DIR_VERBATIM}  -> ABS: {ABS_KNOW_DIR}")
    print(await orchestrate(sys.argv[1], include_docs=True))

if __name__ == "__main__":
    asyncio.run(main())
