# Research Briefing Agent + Verifier

An evidence-grounded research assistant that plans searches, retrieves authoritative Wikipedia evidence, synthesizes a cited research briefing, and verifies factual claims against source text.

---

## 1. Problem Statement

Large Language Models (LLMs) often generate fluent, persuasive answers that contain subtle factual inaccuracies, hallucinated statistics, or phantom citations. When researching complex scientific, technological, or policy topics, users need verified, traceable evidence rather than ungrounded claims.

**Research Briefing Agent + Verifier** solves this problem by combining an **agentic research pipeline** with an **independent factual claim verifier**. Rather than relying on static parametric memory, the system dynamically plans multi-angle Wikipedia searches, retrieves and filters relevant articles, synthesizes a briefing with mandatory inline citations, and performs sentence-by-sentence claim verification against the retrieved text.

---

## 2. Main Features

- **Automated Research Planning**: Decomposes complex topics into distinct, focused search queries covering foundational concepts, real-world evidence, and limitations.
- **Resilient Wikipedia Retrieval**: Leverages Wikipedia's REST APIs using a shared HTTP session with per-article failure isolation (e.g. 404s or timeouts on single articles do not crash the search).
- **LLM-Powered Relevance Filtering**: Evaluates candidate articles against the research topic to eliminate tangential or low-relevance sources.
- **Grounded Briefing Synthesis**: Enforces strict prompt guardrails forbidding outside statistics, unproven causal claims, and speculative figures.
- **Factual Claim Verification**: Extracts checkable claims and audits each claim against evidence text, assigning color-coded verdicts (`supported`, `partially_supported`, `unsupported`, `contradicted`).
- **Citation Integrity & Quotation Auditing**: Detects nonexistent or fabricated citation tags (e.g. citing `[S9]` when only `S1–S3` exist) and validates that supporting excerpts are verbatim substrings of the source text.
- **Streamlit Web Dashboard & CLI**: Full-featured interactive web application with live progress streaming, alongside a command-line interface.

---

## 3. Architecture & Workflow

```mermaid
graph TD
    User([User Topic Input]) --> Planner[1. Research Planner\nMistral open-mistral-nemo]
    Planner -->|4 Focused Search Queries| Retrieval[2. Wikipedia Retrieval\nWikimedia REST APIs]
    Retrieval -->|Candidate Articles| RelFilter[3. Relevance Filter\nMistral Evaluator]
    RelFilter -->|Top Relevant Sources S1..Sn| Synthesizer[4. Briefing Synthesizer\nMistral with Strict Guardrails]
    Synthesizer -->|Cited Briefing [S#]| Verifier[5. Claim Verifier & Auditor\nMistral Fact-Checker]
    Verifier -->|Verdicts, Excerpts, Warnings| UI[6. Streamlit Dashboard & CLI Output]
```

### How the Pipeline Works Step-by-Step

1. **Research Planner (`planner.py`)**: Prompts Mistral to formulate 4 distinct, targeted research questions.
2. **Wikipedia Retrieval (`wikimedia_client.py`)**: Executes searches on Wikipedia's REST search API, fetching lead summary extracts.
3. **Relevance Filter (`relevance_filter.py`)**: Prompts Mistral to review candidates and select up to 6 directly relevant sources.
4. **Briefing Generator (`research_agent.py`)**: Packages retrieved evidence into stable identifiers (`[S1]`, `[S2]`) and prompts Mistral to synthesize an Executive Summary and Key Findings with inline citations.
5. **Claim Verifier (`verifier.py`)**: Extracts checkable factual assertions from the briefing, validates inline citation references, and prompts the verifier to assign verdicts and extract verbatim supporting quotations.
6. **Result Aggregation**: Compiles the plan, sources, briefing, verification results, and any warnings into a structured `ResearchResult`.

---

## 4. APIs Used

| API | Provider | Purpose |
| :--- | :--- | :--- |
| **Mistral API** (`open-mistral-nemo`) | Mistral AI | Powers research question planning, source relevance ranking, briefing synthesis, claim extraction, and factual verification. |
| **Wikimedia REST API** | Wikimedia Foundation | Free, open encyclopedia search endpoint (`search/page`) and summary endpoint (`page/summary/{title}`). Requires no API key. |

---

## 5. RAG & Agentic Workflow

### Retrieval-Augmented Generation (RAG)
Standard LLM prompting generates answers purely from model weights. This project implements **Retrieval-Augmented Generation (RAG)**: factual generation is conditioned strictly on external Wikipedia text injected into the context window. Numerical claims, dates, and causal statements must be substantiated by source excerpts.

### What Makes This "Agentic"?
Unlike a static single-prompt RAG system:
- **Multi-step goal decomposition**: The agent decomposes the research topic into multiple distinct sub-questions.
- **Dynamic retrieval loops**: Queries Wikipedia multiple times, gathers candidate articles, and performs iterative relevance filtering.
- **Independent self-auditing / verification**: The pipeline includes a secondary verification agent that independently cross-checks the synthesized briefing against the ground-truth evidence before presenting findings to the user.

---

## 6. Technology Stack

- **Frontend**: Streamlit 1.28+ (responsive web dashboard, live status indicators, metric cards, expanders).
- **LLM SDK**: Mistral AI Python SDK (`mistralai>=1.0.0`, model: `open-mistral-nemo`).
- **HTTP / Networking**: `requests>=2.28.0` with connection pooling, retries, and failure isolation.
- **Configuration & Environment**: `python-dotenv` and Streamlit Secrets (`st.secrets`).
- **Testing**: Python's built-in `unittest` test runner (46 automated unit and integration tests).

---

## 7. Project Structure

```text
Research-agent/
├── .streamlit/
│   ├── config.toml               # Streamlit server & theme settings
│   └── secrets.toml.example      # Safe placeholder template for Cloud deployment
├── backend/
│   ├── __init__.py
│   ├── mistral_client.py         # Deferred key resolution (env, .env, st.secrets) & LLM caller
│   ├── wikimedia_client.py       # Wikipedia REST API client with article failure isolation
│   ├── planner.py                # Research question generator
│   ├── relevance_filter.py       # Candidate source relevance evaluator
│   ├── verifier.py               # Claim extraction, citation check, and excerpt verification
│   └── research_agent.py         # Core orchestrator returning structured ResearchResult
├── tests/
│   ├── __init__.py
│   ├── test_stage1.py            # Citation validation, excerpt tracing, & claim tests (21 tests)
│   ├── test_stage2.py            # API key resolution & Wikimedia resilience tests (19 tests)
│   └── test_stage3.py            # UI integration & verdict rendering tests (6 tests)
├── app.py                        # Streamlit web application entry point
├── requirements.txt              # Production dependency specifications
├── .gitignore                    # Excludes .env, .venv, local secrets, and caches
└── README.md                     # Documentation and setup guide
```

---

## 8. Prerequisites & Local Setup (Windows)

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Git
- A Mistral AI API key (from [console.mistral.ai](https://console.mistral.ai/))

### 2. Clone the Repository
```powershell
git clone https://github.com/Raghavendra0807/Research-agent.git
cd Research-agent
```

### 3. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 5. Configure Your Mistral API Key Locally
Create a `.env` file in the repository root:
```env
MISTRAL_API_KEY=your_actual_mistral_api_key_here
```
> [!IMPORTANT]
> The `.env` file is excluded in `.gitignore` and must **never** be committed to Git.

---

## 9. Running the Application

### Launch the Streamlit Web Application
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### Run via Command-Line Interface (CLI)
```powershell
.\.venv\Scripts\python.exe -m backend.research_agent
```

### Run the Automated Tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

---

## 10. Deploying on Streamlit Community Cloud

Follow these steps to deploy the application on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. **Push your code to GitHub**:
   Ensure `.env` and `.venv/` are NOT staged or committed (verified by `.gitignore`).
   ```powershell
   git status
   git add app.py backend/ tests/ .streamlit/ requirements.txt README.md .gitignore
   git commit -m "Prepare Research Briefing Agent + Verifier for deployment"
   git push origin main
   ```
2. **Create New App on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **New app**.
   - Select your repository: `Raghavendra0807/Research-agent`.
   - Set **Branch**: `main`.
   - Set **Main file path**: `app.py`.
3. **Configure Secrets**:
   - In the deployment setup (or via **App Settings -> Secrets**), add your Mistral API key in TOML format:
     ```toml
     MISTRAL_API_KEY = "your_actual_mistral_api_key_here"
     ```
   - Click **Save**.
4. **Deploy**:
   - Click **Deploy!**. Streamlit Cloud will install dependencies from `requirements.txt` and launch `app.py`.

---

## 11. Known Limitations

- **Wikipedia Scope**: Research evidence is limited to topics documented on English Wikipedia. Topics requiring proprietary data or very recent breaking news may have limited or absent sources.
- **Model Summarization**: Briefings are synthesized by a language model; while guardrails strictly discourage speculation, users should review cited source links for critical decisions.
- **Evidence Verification vs. Absolute Truth**: Verifier verdicts evaluate whether a claim is substantiated by the retrieved text excerpt. A `supported` verdict means the text supports the statement, not that the statement represents absolute external truth.
- **API Rate Limits**: The Mistral API and Wikipedia REST endpoints are subject to network latency and standard rate limits.