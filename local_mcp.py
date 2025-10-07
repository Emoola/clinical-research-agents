import os, asyncio, json
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

"""
Go to root and start with:
npx @modelcontextprotocol/server-filesystem knowledge
"""

BASE = os.path.dirname(os.path.abspath(__file__))
KNOW = os.getenv("KNOWLEDGE_DIR", os.path.join(BASE, "knowledge"))
ROOT = os.path.abspath(KNOW)

async def main():
    print("ROOT:", ROOT)
    args = ["@modelcontextprotocol/server-filesystem","--stdio","--root",ROOT,"--allow-write","false"]
    npm_path = os.path.join(os.path.dirname(os.popen('where node').read().strip()), 'npx.cmd')
    cmd = os.getenv("MCP_FS_CMD", npm_path)
    print(f"Using command: {cmd}")
    async with stdio_client(StdioServerParameters(command=cmd, args=args)) as (r,w):
        async with ClientSession(r,w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            for tool in ["list_directory","filesystem.list_directory","fs.list","list","dir","files.list"]:
                try:
                    res = await s.call_tool(tool, arguments={"path": ".", "recursive": False})
                except Exception:
                    continue
                if res:
                    t = ""
                    if getattr(res,"structuredContent",None): t = json.dumps(res.structuredContent, indent=2)
                    else:
                        for c in res.content:
                            if hasattr(c,"text"): t += c.text
                    print(f"\nLIST via {tool} ->\n{t[:600]}")
                    break
            for readname in ["read_file","filesystem.read_file","fs.read","read","file.read","files.read"]:
                for variant in ["discharge_policy.txt","./discharge_policy.txt","/discharge_policy.txt"]:
                    try:
                        res = await s.call_tool(readname, arguments={"path": variant})
                    except Exception:
                        continue
                    if res:
                        txt = ""
                        if getattr(res,"structuredContent",None): txt = json.dumps(res.structuredContent, indent=2)
                        else:
                            for c in res.content:
                                if hasattr(c,"text"): txt += c.text
                        print(f"\nREAD via {readname} path={variant} ->\n{txt}")
                        return
            print("\nCould not read discharge_policy.txt via MCP. Check tool names and path.")

asyncio.run(main())
