# 🎯 AICTE Interview Trainer Agent

> **AI-powered interview preparation coach** — Powered by **IBM Granite (watsonx.ai)** + **RAG (Retrieval-Augmented Generation)**

[![IBM Granite](https://img.shields.io/badge/IBM%20Granite-watsonx.ai-0062ff)](https://www.ibm.com/watsonx)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io)
[![RAG](https://img.shields.io/badge/Architecture-RAG-green)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)
[![License](https://img.shields.io/badge/License-AICTE%20Challenge-orange)](https://aicte-india.org)

---

## 🌟 Overview

The **AICTE Interview Trainer Agent** is a production-quality, end-to-end AI system that prepares job seekers for technical and HR interviews through:

| Feature | Description |
|---|---|
| 📋 **Tailored Question Generation** | Role-specific questions matched to your exact experience level, skills, and target company |
| ✅ **AI Answer Evaluation** | Scored feedback (1–10) with strengths, improvement areas, and model answers |
| 📅 **7-Day Prep Strategy** | Personalized day-by-day preparation roadmap |
| 🎭 **Behavioral Coach** | STAR-method questions for 10+ competency scenarios with scoring rubrics |
| 💡 **Improvement Tips** | 7 actionable, role-specific tips to maximize interview success |
| 📄 **Resume Parsing** | Auto skill extraction and experience detection from PDF/DOCX resumes |
| 📊 **Session Export** | Download all outputs as TXT or JSON |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AICTE Interview Trainer Agent v2.0                │
├─────────────────────────┬────────────────────────────────────────────┤
│   Streamlit Web UI      │            CLI (Typer + Rich)              │
│   (app.py) — 6 tabs     │            (cli.py)                        │
├─────────────────────────┴────────────────────────────────────────────┤
│                     Agent Orchestrator                               │
│                     (src/agent.py)                                   │
├──────────────────────────────┬───────────────────────────────────────┤
│   Interview Engine           │   User Profile Builder               │
│   (src/interview_engine.py)  │   (src/user_profile.py)              │
│   • Question Generation      │   • Resume Text Extraction           │
│   • Answer Evaluation        │   • Skill Auto-Detection             │
│   • Prep Strategy            │   • Experience Level Classification  │
│   • Behavioral Assessment    │   • Profile Context Builder          │
│   • Improvement Tips         │                                      │
├──────────────────────────────┼───────────────────────────────────────┤
│   IBM Granite LLM            │   RAG Pipeline                       │
│   (src/granite_llm.py)       │   (src/rag_pipeline.py)             │
│   • granite-3-8b-instruct    │   • ChromaDB (semantic search)       │
│   • watsonx.ai REST API      │   • Sentence-Transformers embeddings │
│   • Native chat templates    │   • Keyword fallback retrieval       │
│   • 6 specialized prompts    │   • Document ingestion (MD/PDF/DOCX) │
│   • Rich demo fallback       │   • In-memory fallback               │
├──────────────────────────────┴───────────────────────────────────────┤
│                       Knowledge Base                                 │
│     data/knowledge_base/ — 7 role-specific markdown guides          │
│     software_engineering · data_science · devops_cloud              │
│     behavioral_interview · product_management · hr_recruiter        │
│     ibm_company_guide                                               │
└──────────────────────────────────────────────────────────────────────┘
```

### RAG Flow
```
Candidate Profile (role + skills + experience)
         │
         ▼
   Build Query ──► ChromaDB Vector Search ──► Top-K Relevant Chunks
         │                                            │
         └────────────────────────────────────────────┘
                                │
                                ▼
               IBM Granite Prompt (System + Context + Candidate Profile)
                                │
                                ▼
               Generated Interview Content (Questions / Evaluation / Strategy)
```

---

## 🚀 Quick Start

### 1. Prerequisites
```
Python 3.9+ (Python 3.11 recommended for full compatibility)
```

### 2. Install Core Dependencies
```bash
cd PROJECT

# Option A: Minimal install (demo mode — works without IBM credentials)
pip install streamlit python-dotenv pydantic loguru rich typer pypdf python-docx pdfplumber requests

# Option B: Full install including IBM Granite
pip install -r requirements.txt
pip install ibm-watsonx-ai

# Option C: Full install with RAG (requires C++ build tools)
pip install -r requirements.txt
pip install ibm-watsonx-ai chromadb sentence-transformers
```

### 3. Configure IBM Credentials (Optional — Demo Mode works without this)
```bash
cp .env.example .env
# Edit .env and add your credentials:
# WATSONX_API_KEY=your_ibm_cloud_api_key
# WATSONX_PROJECT_ID=your_watsonx_project_id
```

> 🆓 **Get free IBM Cloud credentials**:
> 1. Register at https://cloud.ibm.com/registration
> 2. Create a watsonx.ai project at https://dataplatform.cloud.ibm.com
> 3. Generate API key at https://cloud.ibm.com/iam/apikeys

### 4. Launch the Web UI
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 5. Use the CLI (Alternative)
```bash
# Full preparation pipeline
python cli.py prep --name "Alice Kumar" --role "Data Scientist" --level "Mid-Level" --years 4

# Generate questions only
python cli.py questions --name "Bob Singh" --role "DevOps Engineer" --num 15

# Interactive answer evaluator
python cli.py evaluate --name "Charlie Patel" --role "Software Engineer"

# Check agent status
python cli.py status
```

---

## 📁 Project Structure

```
PROJECT/
├── app.py                          # Streamlit web application (6 tabs)
├── cli.py                          # Command-line interface (4 commands)
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration management (Pydantic)
│   ├── granite_llm.py              # IBM Granite LLM wrapper + 6 prompt templates
│   ├── rag_pipeline.py             # RAG: ingest → embed → store → retrieve
│   ├── user_profile.py             # Profile builder + resume parser + skill extractor
│   ├── interview_engine.py         # Core interview coaching engine
│   └── agent.py                    # Top-level agent orchestrator (singleton)
│
└── data/
    ├── knowledge_base/             # RAG document sources (7 files)
    │   ├── software_engineering.md # DS&A, system design, OOP, industry expectations
    │   ├── data_science.md         # ML algorithms, statistics, IBM watsonx questions
    │   ├── devops_cloud.md         # CI/CD, Kubernetes, IBM Cloud, monitoring
    │   ├── behavioral_interview.md # STAR method, 30 behavioral questions, competencies
    │   ├── product_management.md   # PM frameworks, roadmapping, metrics
    │   ├── hr_recruiter_guide.md   # HR questions, salary negotiation, ATS tips
    │   └── ibm_company_guide.md    # IBM Granite, watsonx.ai, IBM interview process
    └── vectorstore/                # ChromaDB persistent store (auto-created)
```

---

## 🤖 IBM Granite Integration

This project **mandatorily** uses **IBM Granite** via **watsonx.ai (IBM Cloud Lite)**:

| Component | IBM Service | Details |
|---|---|---|
| Language Model | IBM Granite 3-8B Instruct | `ibm/granite-3-8b-instruct` (default) |
| Platform | IBM watsonx.ai | IBM Cloud Lite (free tier) |
| SDK | `ibm-watsonx-ai` | Python SDK v1.1.2+ |
| Prompt Format | Granite Native Chat | `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>` |
| Embeddings | Sentence-Transformers | `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB | Local persistent storage |

### Prompt Engineering — 6 Specialized Templates
All prompts use IBM Granite's native chat format optimized for Granite 3.x:

| Template | Purpose |
|---|---|
| `QUESTION_GENERATION_TEMPLATE` | Generates N tailored questions with assessment criteria |
| `ANSWER_EVALUATION_TEMPLATE` | Scores answers 1-10 with structured feedback |
| `PREPARATION_STRATEGY_TEMPLATE` | Creates 7-day day-by-day preparation plan |
| `BEHAVIORAL_ASSESSMENT_TEMPLATE` | STAR questions with scoring rubric per scenario |
| `IMPROVEMENT_TIPS_TEMPLATE` | 7 actionable, role-specific tips per candidate |

### Demo Mode
Without IBM credentials, the agent runs in **Demo Mode** with rich, structured mock responses that demonstrate the full feature set. This allows complete exploration of the UI without IBM Cloud access.

---

## 🎯 Features In Depth

### 📋 Question Generation
- **10–20 questions** per session (configurable)
- **4 categories**: Technical, Behavioral (STAR), Situational, HR/Culture-fit
- **3 difficulty levels**: Easy, Medium, Hard — calibrated to experience level
- **Assessment criteria**: What each question evaluates
- **Model answer frameworks**: Structural guidance for ideal responses
- **Target company tailoring**: Questions adjusted for specific companies (IBM, Google, etc.)
- **Interview type modes**: Full Stack / Technical Only / Behavioral Only / HR Only

### ✅ Answer Evaluation
- **Score 1–10** with color-coded visual progress bar
- **Strength analysis**: What the candidate did well
- **Improvement specifics**: Exact gaps with actionable fixes
- **Model answer**: A demonstrably better version of the response
- **Follow-up anticipation**: 2-3 likely follow-up questions

### 📅 7-Day Preparation Strategy
- **Daily breakdowns**: Morning / Afternoon / Evening tasks with time allocations
- **Role-specific topics**: 8-10 study areas with priority (High/Medium/Low)
- **Resource curation**: Books, platforms, IBM-specific learning
- **Mock interview scheduling**: When and how to run simulation sessions
- **5 critical pitfalls**: Role-specific mistakes to avoid
- **Confidence building**: Practical exercises for interview anxiety

### 🎭 Behavioral Coach
- **10+ pre-built scenarios**: Leadership, conflict, failure, pressure, innovation, ethics, and more
- **Custom scenario support**: Enter any competency scenario
- **STAR answer templates**: Fill-in-the-blanks structure for each scenario
- **Scoring rubric**: 5/10 vs 9/10 answer quality breakdown table
- **Red flag awareness**: Specific interviewer warning signs to avoid

### 💡 Improvement Tips
- **7 high-impact tips** per candidate (role + level specific)
- Each tip includes: The Problem → The Fix → Quick Practice (5-min exercise)
- Universal success principles always available as expandable reference cards

### 📄 Resume Integration
- **PDF, DOCX, TXT** resume parsing
- **Automatic skill extraction**: 100+ skills across 7 categories matched via NLP
- **Experience level auto-detection**: Heuristic year extraction from resume text
- **Resume ingested into RAG**: Enables hyper-personalized question generation

---

## ⚙️ Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `WATSONX_API_KEY` | IBM Cloud API Key | *(demo mode if unset)* |
| `WATSONX_PROJECT_ID` | watsonx.ai Project ID | *(demo mode if unset)* |
| `WATSONX_URL` | watsonx.ai endpoint | `https://us-south.ml.cloud.ibm.com` |
| `GRANITE_MODEL_ID` | Granite model identifier | `ibm/granite-3-8b-instruct` |
| `GRANITE_MAX_TOKENS` | Max response tokens | `1500` |
| `GRANITE_TEMPERATURE` | Generation temperature | `0.7` |
| `GRANITE_TOP_P` | Nucleus sampling | `0.9` |
| `GRANITE_TOP_K` | Top-K sampling | `50` |
| `GRANITE_REP_PENALTY` | Repetition penalty | `1.1` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./data/vectorstore` |
| `EMBEDDING_MODEL` | Sentence-BERT model | `all-MiniLM-L6-v2` |
| `KNOWLEDGE_BASE_DIR` | KB documents directory | `./data/knowledge_base` |

---

## 🧪 Demo Mode vs Live Mode

| Feature | Demo Mode | Live Mode (IBM Granite) |
|---|---|---|
| **Setup** | Zero config needed | Requires IBM Cloud account |
| **Question Quality** | Rich structured mock responses | Fully personalized AI generation |
| **Answer Evaluation** | Structured demo feedback | Real-time scored AI feedback |
| **Preparation Strategy** | Comprehensive demo plan | Personalized to candidate + company |
| **Behavioral Coach** | Structured demo assessment | Scenario-specific AI analysis |
| **Resume Personalization** | Profile-based (local) | Full RAG + AI personalization |

---

## 📊 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **AI / LLM** | IBM Granite 3-8B (watsonx.ai) | All text generation |
| **RAG** | ChromaDB + custom pipeline | Knowledge retrieval |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Semantic similarity |
| **Web UI** | Streamlit | Interactive web interface |
| **CLI** | Typer + Rich | Terminal interface |
| **Resume Parsing** | PyPDF + python-docx | Document extraction |
| **Config** | Pydantic v2 + python-dotenv | Type-safe configuration |
| **Logging** | Loguru | Structured application logging |
| **Data Models** | Pydantic v2 | UserProfile, QuestionSet, Evaluation |

---

## 🔒 Graceful Degradation

The system is designed to function at every dependency level:

```
Level 1 — Core (always works):
  Streamlit UI + Pydantic + python-dotenv

Level 2 — Resume parsing:
  + pypdf + python-docx

Level 3 — Demo AI (rich structured responses):
  + loguru (no IBM credentials needed)

Level 4 — Live AI:
  + ibm-watsonx-ai + IBM Cloud credentials

Level 5 — Full RAG:
  + chromadb + sentence-transformers
```

---

## 📄 License

Developed as part of the **AICTE AI Challenge** using mandatory IBM Cloud Lite services.

---

*Made with ❤️ using IBM Granite · watsonx.ai · IBM Cloud Lite*
