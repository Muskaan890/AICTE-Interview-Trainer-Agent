# 🎯 AICTE Interview Trainer Agent

> **AI-powered interview preparation coach** — Powered by **IBM Granite (watsonx.ai)** + **RAG (Retrieval-Augmented Generation)**

[![IBM Granite](https://img.shields.io/badge/IBM%20Granite-watsonx.ai-0062ff)](https://www.ibm.com/watsonx)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io)
[![RAG](https://img.shields.io/badge/Architecture-RAG-green)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

---

## 🌟 Overview

The **AICTE Interview Trainer Agent** is an end-to-end AI-driven system that prepares job seekers for technical and HR interviews through:

- 🎯 **Personalised question generation** — tailored to name, role, experience level, and skills
- 📊 **Answer evaluation & scoring** — AI-scored feedback using IBM Granite
- 📅 **7-day preparation strategy** — day-wise personalized study plan
- 🎭 **Behavioral coaching** — STAR-method questions for any competency scenario
- 💡 **Improvement tips** — targeted guidance to boost interview performance
- 📄 **Resume parsing** — automatic skill extraction from PDF/DOCX resumes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Interview Trainer Agent                        │
├─────────────────────┬───────────────────────────────────────────┤
│   Streamlit UI      │           CLI (Typer/Rich)                │
│   (app.py)          │           (cli.py)                        │
├─────────────────────┴───────────────────────────────────────────┤
│                    Agent Orchestrator                           │
│                    (src/agent.py)                               │
├──────────────────────────────┬──────────────────────────────────┤
│   Interview Engine           │   User Profile Builder           │
│   (src/interview_engine.py)  │   (src/user_profile.py)         │
├──────────────────────────────┼──────────────────────────────────┤
│   IBM Granite LLM            │   RAG Pipeline                  │
│   (src/granite_llm.py)       │   (src/rag_pipeline.py)         │
│   watsonx.ai API             │   ChromaDB + Sentence-BERT      │
├──────────────────────────────┴──────────────────────────────────┤
│                    Knowledge Base                               │
│         data/knowledge_base/*.md (role-specific guides)        │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Flow
```
User Query (role + skills)
        │
        ▼
  Embed Query ──► ChromaDB Vector Search
        │                  │
        │          Relevant Chunks
        │                  │
        └─────────────────►│
                           ▼
              IBM Granite Prompt (Context + Query)
                           │
                           ▼
              Generated Interview Content
```

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
Python 3.9+
```

### 2. Install Dependencies
```bash
cd PROJECT
pip install -r requirements.txt
```

### 3. Configure IBM Credentials (Optional — Demo mode works without this)
```bash
cp .env.example .env
# Edit .env and set:
# WATSONX_API_KEY=your_ibm_cloud_api_key
# WATSONX_PROJECT_ID=your_watsonx_project_id
```

> **Get free IBM Cloud credentials**: https://cloud.ibm.com/registration  
> **watsonx.ai Free Tier**: https://dataplatform.cloud.ibm.com/

### 4. Launch Web UI
```bash
streamlit run app.py
```

### 5. Use CLI (Alternative)
```bash
# Full preparation pipeline
python cli.py prep --name "Alice Kumar" --role "Data Scientist" --level "Mid-Level" --years 4

# Generate questions only
python cli.py questions --name "Bob Singh" --role "DevOps Engineer" --num 15

# Interactive answer evaluator
python cli.py evaluate --name "Charlie" --role "Software Engineer"

# Check agent status
python cli.py status
```

---

## 📁 Project Structure

```
PROJECT/
├── app.py                          # Streamlit web application
├── cli.py                          # Command-line interface
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── granite_llm.py              # IBM Granite LLM wrapper
│   ├── rag_pipeline.py             # RAG: ingest, embed, retrieve
│   ├── user_profile.py             # Profile builder + resume parser
│   ├── interview_engine.py         # Core interview logic
│   └── agent.py                    # Top-level agent orchestrator
│
└── data/
    ├── knowledge_base/             # Interview guides (RAG source)
    │   ├── software_engineering.md
    │   ├── data_science.md
    │   ├── behavioral_interview.md
    │   ├── devops_cloud.md
    │   ├── product_management.md
    │   └── ibm_company_guide.md
    └── vectorstore/                # ChromaDB persistent store (auto-created)
```

---

## 🤖 IBM Granite Integration

This project mandatorily uses **IBM Granite** via **watsonx.ai** (IBM Cloud Lite):

| Component | IBM Service |
|---|---|
| Language Model | IBM Granite 13B Instruct v2 |
| Platform | IBM watsonx.ai (IBM Cloud Lite) |
| SDK | `ibm-watsonx-ai` Python SDK |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (local persistent) |

### IBM Granite Models Used
- `ibm/granite-13b-instruct-v2` — Default; balanced performance
- `ibm/granite-3-8b-instruct` — Lighter, faster responses
- `ibm/granite-3b-code-instruct` — For code-related interview Qs

### Prompt Architecture
All prompts use IBM Granite's native `<|system|>`, `<|user|>`, `<|assistant|>` chat template format for optimal performance.

---

## 🎯 Features In Depth

### Question Generation
- Role-specific technical questions (Software, Data Science, DevOps, PM, etc.)
- Behavioral questions with STAR method guidance
- Situational / scenario-based questions
- HR and culture-fit questions
- Difficulty levels: Easy / Medium / Hard
- Model answer hints and evaluation criteria

### Answer Evaluation
- AI-powered scoring (1-10) with justification
- Strength analysis
- Specific improvement suggestions
- Model answer with metrics/structure
- Follow-up questions an interviewer would ask

### Preparation Strategy
- Personalized 7-day study plan
- Day-by-day task breakdown
- Topic-specific resource recommendations
- Mock interview scheduling guidance
- Common pitfall awareness

### Behavioral Assessment
- STAR-method question generation for any competency
- 10+ pre-built scenarios (leadership, conflict, failure, pressure, etc.)
- Custom scenario support
- Red flag awareness guide

### Resume Integration
- PDF, DOCX, and plain text resume parsing
- Automatic skill extraction via NLP keyword matching
- Experience level auto-detection
- Resume ingested into RAG for hyper-personalized questions

---

## ⚙️ Configuration

| Variable | Description | Default |
|---|---|---|
| `WATSONX_API_KEY` | IBM Cloud API Key | *(demo mode if unset)* |
| `WATSONX_PROJECT_ID` | watsonx.ai Project ID | *(demo mode if unset)* |
| `WATSONX_URL` | watsonx.ai endpoint | `https://us-south.ml.cloud.ibm.com` |
| `GRANITE_MODEL_ID` | Model identifier | `ibm/granite-13b-instruct-v2` |
| `GRANITE_MAX_TOKENS` | Max response tokens | `1024` |
| `GRANITE_TEMPERATURE` | Sampling temperature | `0.7` |
| `CHROMA_PERSIST_DIR` | Vector store path | `./data/vectorstore` |
| `EMBEDDING_MODEL` | Sentence-BERT model | `all-MiniLM-L6-v2` |
| `KNOWLEDGE_BASE_DIR` | KB documents path | `./data/knowledge_base` |

---

## 🧪 Demo Mode

Without IBM credentials, the agent runs in **Demo Mode** which returns rich, realistic mock responses. This allows full exploration of the UI and functionality without requiring IBM Cloud access.

To activate IBM Granite live mode:
1. Register at https://cloud.ibm.com/registration (free)
2. Create a watsonx.ai project at https://dataplatform.cloud.ibm.com
3. Generate an API key at https://cloud.ibm.com/iam/apikeys
4. Update `.env` with your credentials

---

## 📊 Technology Stack

| Layer | Technology |
|---|---|
| AI/LLM | IBM Granite (watsonx.ai) |
| RAG | LangChain + ChromaDB |
| Embeddings | Sentence-Transformers |
| Web UI | Streamlit |
| CLI | Typer + Rich |
| Resume Parsing | PyPDF + python-docx |
| Config | Pydantic + python-dotenv |
| Logging | Loguru |

---

## 📄 License

This project is developed as part of the **AICTE AI Challenge** using IBM Cloud Lite services.

---

*Made with ❤️ using IBM Granite · watsonx.ai · IBM Cloud*
