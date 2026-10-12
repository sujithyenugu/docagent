# DocumentAgent

**Agentic Document & Form Extraction for India's Tier 2/3 Cities**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-TBD-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Problem Statement

Manual, handwritten bookkeeping dominates small businesses and government offices in India's tier 2 and tier 3 cities. This creates real problems:

- **Error-prone**: Handwritten ledgers, invoices, and forms are transcribed manually, introducing mistakes at every step.
- **Slow**: Data entry by hand or retyping scanned documents is a significant time sink.
- **Inaccessible automation**: Existing OCR and AI extraction tools are either too expensive, require cloud connectivity, or are not designed for Indian document formats and handwriting styles.

DocumentAgent addresses these gaps with a practical, privacy-conscious extraction pipeline that works for both internet-connected and offline environments.

---

## Solution

DocumentAgent is a dual-mode agentic document extraction system:

1. **Cloud path (Landing AI AIDE)**: Uses [Landing AI's AIDE API](https://va.landing.ai/) for high-accuracy structured extraction from scanned or photographed documents — invoices, ledgers, ID documents, government forms.

2. **Local path (Qwen2.5-VL-3B via Apple MLX)**: An on-device pipeline running Qwen2.5-VL-3B (VLM variant) directly in-process via [Apple MLX](https://github.com/ml-explore/mlx) — no separate server to install or run, no external API calls during inference. Apple Silicon Mac only.

3. **Web demo (Streamlit)**: Developers can explore the interface freely; actual agent extraction requires a Landing AI API key or local model setup.

---

## Key Features

- Extract **text, tables, and key-value pairs** from scanned or photographed Indian documents
- Supports both **handwritten and printed** input
- **Dual-mode inference**: cloud API (Landing AI AIDE) or local model (Qwen2.5-VL-3B via embedded Apple MLX — no setup)
- Interactive **Streamlit web demo** and **Jupyter notebook** for feasibility testing
- **AWS-hosted production backend**: S3, Lambda, Bedrock, Strands Agents
- **Privacy controls**: choose between cloud processing or fully local inference — no documents sent to external servers in local mode
- No document storage by default — opt-in S3 toggle only

---

## Architecture

A dual-mode pipeline routes documents through either the Landing AI AIDE cloud API or a local quantized Qwen2.5-VL model, then surfaces structured output via a Streamlit interface backed by an optional AWS production layer.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full diagram, technical stack, privacy model, pricing tiers, and hardware prerequisites.

---

## Roadmap

Planned work is tracked as GitHub Issues. See the [Issues tab](https://github.com/knightinout/DocumentAgent/issues) for the full backlog, organised by phase label (`phase1-web-ui-mvp`, `phase2-local-documentagent`, `phase3-production-deploy`).

### V1 — Document Extraction (current)

Upload any document (PDF, image) and get clean structured Markdown output. Runs fully locally via Qwen2.5-VL-3B through embedded Apple MLX — no data leaves the device, no separate app to install.

### V2 — Privacy-First Form Fill from Multiple ID Documents

**Goal:** User uploads multiple identity documents; the app extracts fields from each and auto-fills a target form, producing a ready-to-submit output — without any document leaving the machine.

**Primary use case — Lease Agreement (India):**

1. User uploads **Aadhaar card** and **driving licence** (tenant + landlord)
2. Local model extracts: name, address, DOB, ID numbers from each document
3. App maps extracted fields onto a **standard lease agreement template**
4. User reviews the pre-filled form and confirms
5. With consent, the **signature extracted from the ID document** is placed on the agreement
6. Final signed PDF is generated — ready for submission at the sub-registrar office

**Why local model matters here:** ID documents contain PAN, Aadhaar, address, photo, and signature. Running inference locally ensures this data is never transmitted to any external server. Privacy is not a feature — it is the architecture.

**Planned UI additions for V2:**
- Multi-file upload panel (drag multiple documents at once)
- Field extraction review table (editable before form fill)
- Form template selector (lease, rent receipt, affidavit, etc.)
- Signature placement preview with consent checkbox
- Final PDF download

---

## Prerequisites

**All paths:**
- Python 3.10+

**Cloud extraction (Phase 1):**
- [Landing AI API key](https://va.landing.ai/) — free tier available

**Local extraction (Phase 2):**
- Apple Silicon Mac (macOS 14+) — no other setup; the model (~2.5 GB) downloads automatically on first extraction

**AWS production backend (Phase 3):**
- AWS account with access to S3, Lambda, Bedrock, and Secrets Manager
- Use **us-east-1** or **us-west-2** — Bedrock is not available in all regions

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/knightinout/DocumentAgent.git
cd DocumentAgent
```

### 2. Set up API key

```bash
cp .env.example .env
```

Open `.env` and add your key (get one at [va.landing.ai](https://va.landing.ai/)):

```
VISION_AGENT_API_KEY=your_key_here
```

### 3. Install and run

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a document or pick a sample from the `samples/` folder, select a model, and click **Extract**.

---

## Phase 2 — Local Model (Embedded Apple MLX)

The **Local** mode runs Qwen2.5-VL-3B directly in the same Python process via [Apple MLX](https://github.com/ml-explore/mlx) — no separate app to install or keep running, no API key, no port to configure.

### Requirements
- Apple Silicon Mac (M1 or later), macOS 14+
- ~2.5 GB free disk space (downloaded automatically, once)

### Setup steps

There isn't one — it just works:

1. Open the **Local** page in the app and click **Extract**.
2. On first use, the model downloads automatically (~2.5 GB; a few minutes depending on your connection). Subsequent extractions are fast, since the model stays loaded in memory for the life of the process.

To use a different MLX vision model, set `LOCAL_MODEL_REPO` in `.env` to any `mlx-community` vision-language repo id.

> Local mode requires Apple Silicon macOS. On other platforms (Linux, Windows, Intel Mac, or inside the Docker image), the Local page clearly reports it's unavailable — use Cloud mode instead.

---

## Building the Mac App

DocumentAgent can be packaged as a double-clickable `.app` — no terminal or `pip install` needed to run it:

```bash
bash packaging/build_mac_app.sh
open dist/DocumentAgent.app   # first launch: right-click → Open (ad-hoc signed, unnotarized)
```

The build script creates a self-contained bundle (app source + a real Python venv with Streamlit/MLX installed) under `dist/DocumentAgent.app`. Launching it starts the server in the background and opens your browser to it automatically; quitting stops the server. Model weights are not bundled — they download to your Hugging Face cache on first extraction, same as running from source.

---

## Running the Notebook

`notebook/DocumentAgent_notebook.ipynb` demonstrates DocumentAgent's AIDE-powered extraction with Tesseract shown as a legacy OCR baseline.

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebook/DocumentAgent_notebook.ipynb
```

A valid `VISION_AGENT_API_KEY` in your `.env` is required for the Landing AI cells. The Tesseract cells run fully offline.

> **Note:** Tesseract must be installed at the OS level.
> - Linux: `sudo apt install tesseract-ocr`
> - macOS: `brew install tesseract`
> - Windows: [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Phase 3 — AWS Production Setup

### Automated setup (recommended)

All AWS resources are created by a single script — no console required:

```bash
cp .env.example .env   # add VISION_AGENT_API_KEY + AWS_ACCESS_KEY_ID/SECRET
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_bedrock.py   # confirm model access before proceeding
python scripts/setup_aws.py       # creates all 7 resources, writes IDs to .env
```

The script creates: S3 bucket, Secrets Manager secret, IAM roles, Lambda function,
S3 event trigger, OpenSearch Serverless collection + vector index, and Bedrock
Knowledge Base. It is idempotent — safe to re-run if interrupted.

### Query documents with the Strands Agent

```bash
python agent.py "What is the total amount on the invoice?"
# or interactive:
python agent.py
```

### Agent model options

| Stage | Set `BEDROCK_MODEL_ID` in `.env` | Notes |
|---|---|---|
| Testing (default) | `amazon.nova-lite-v1:0` | No Anthropic payment gate, ~$0.003/10 queries |
| Pre-demo | `anthropic.claude-haiku-4-5-20251001-v1:0` | Fast, high quality, ~$0.05/10 queries |
| Demo | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Best accuracy, ~$0.17/10 queries |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline diagram, cost breakdown, and IAM policy details.

---

## Evaluating Extraction Quality

`benchmarks/eval_models.py` compares any number of local MLX models and/or
Landing AI ADE variants side by side on the same document dataset, so a
model swap is judged on this project's actual documents (Indian handwriting,
GST invoices, mixed Hindi/English forms) rather than general-purpose
benchmarks:

```bash
# Built-in candidate set: current default + two Qwen2.5-VL sizes + AIDE
python benchmarks/eval_models.py \
    --dataset-dir benchmarks/dataset \
    --api-key YOUR_LANDING_AI_KEY \
    --defaults

# Or specify your own model list
python benchmarks/eval_models.py \
    --dataset-dir benchmarks/dataset \
    --api-key YOUR_LANDING_AI_KEY \
    --model gemma-4b=local:mlx-community/gemma-3-4b-it-4bit \
    --model qwen7b=local:mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
    --model aide-mini=cloud:dpt-2-mini

# Also include a model served through a running LM Studio instance —
# benchmark-only; LM Studio is not a runtime dependency of the app itself
python benchmarks/eval_models.py \
    --dataset-dir benchmarks/dataset \
    --model qwen3b=local:mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
    --model qwen-lmstudio=lmstudio:qwen2.5-vl-7b-instruct \
    --lmstudio-base-url http://localhost:1234/v1
```

Writes `benchmarks/model_comparison.json` (raw per-document results) and
`benchmarks/model_comparison.md` (ranked summary + per-category breakdown).
Each local model is downloaded on first use (same as normal app usage) — the
7B Qwen2.5-VL variant is ~4.5 GB. An `lmstudio:` spec instead calls a model
already loaded in [LM Studio](https://lmstudio.ai/)'s local server (OpenAI-
compatible REST API) — useful for comparing a model only available as a GGUF
build, but purely a benchmarking convenience: the app's own Local mode always
runs embedded via Apple MLX (see [ARCHITECTURE.md](ARCHITECTURE.md)).
`--strict` exits 1 if every local model trails the best cloud model by more
than 15 percentage points, for CI gating.

To switch the app's local model once you've picked a winner, set
`LOCAL_MODEL_REPO` in `.env` (see `.env.example`) — no code changes needed.

---

## Running Tests

```bash
source .venv/bin/activate
pip install pytest pytest-cov
pytest tests/ -v --cov=storage --cov=handler --cov=local_model --cov-report=term-missing
```

All tests are mocked — no AWS credentials or local model download required.

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Open a pull request

---

## License

License to be decided. See [LICENSE](LICENSE) once finalised.

---

## References

- [Landing AI AIDE](https://va.landing.ai/)
- [Strands Agents](https://github.com/strands-agents/strands-agents)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) / [Gemma Model Family](https://ai.google.dev/gemma)
- [Apple MLX](https://github.com/ml-explore/mlx) / [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
- [Streamlit](https://streamlit.io)
