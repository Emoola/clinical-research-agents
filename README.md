# Clinical Evidence Copilot (Azure AI Foundry + Semantic Kernel + MCP)

**Goal:** Run a local, multi‑agent healthcare demo that calls your **Azure AI Foundry** model and uses **published MCP servers** to gather evidence from PubMed, ClinicalTrials.gov, the web (guidelines), and a local policy folder.

## Quick Start

```bash
pip install "semantic-kernel>=1.31.0" azure-identity
npm i -g @cyanheads/pubmed-mcp-server clinicaltrialsgov-mcp-server @sylphlab/tools-fetch-mcp @modelcontextprotocol/server-filesystem
export AZUREAI_PROJECT_ENDPOINT="https://<service>.services.ai.azure.com/api/projects/<project>"
export AZUREAI_MODEL_DEPLOYMENT="<your-chat-model-deployment>"
python run_all.py
```

## Structure

```
clinical-evidence-copilot/
├── app/
│   ├── agents.py
│   ├── mcp_servers.py
│   └── settings.py
├── prompts/
│   ├── literature.md
│   ├── trials.md
│   ├── guidelines.md
│   ├── safety.md
│   ├── docs.md
│   └── orchestrator.md
├── knowledge/
│   ├── hospital_policy.txt
│   ├── obesity_clinic_policy.txt
│   ├── glp1_local_protocol.txt
│   └── telehealth_policy.txt
├── run_all.py
└── README.md
```

## Branching & Flow

```
Orchestrator → (parallel) Literature | Trials | Guidelines | Safety | Docs → Orchestrator → Final answer with Sources
```

## Customize
- Change prompts in `prompts/` without touching code.
- Pass a different question to `run_all.py` as CLI args.
- Point to another policy folder via `KNOWLEDGE_DIR` env.
