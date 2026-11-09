"""
config.py – Central configuration loader for the Interview Trainer Agent.
Reads from environment variables / .env file and exposes typed settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Absolute path to the PROJECT folder (parent of the src/ directory).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()  # fallback to system env

# Default absolute paths for data directories
_DEFAULT_KB_DIR     = str(_PROJECT_ROOT / "data" / "knowledge_base")
_DEFAULT_VECTOR_DIR = str(_PROJECT_ROOT / "data" / "vectorstore")


class WatsonxConfig(BaseModel):
    api_key:        str   = Field(default_factory=lambda: os.getenv("WATSONX_API_KEY", ""))
    project_id:     str   = Field(default_factory=lambda: os.getenv("WATSONX_PROJECT_ID", ""))
    url:            str   = Field(default_factory=lambda: os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"))
    # Use the latest Granite 3.3 model; fall back to 13b-instruct for older SDKs
    model_id:       str   = Field(default_factory=lambda: os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct"))
    max_new_tokens: int   = Field(default_factory=lambda: int(os.getenv("GRANITE_MAX_TOKENS", "1500")))
    temperature:    float = Field(default_factory=lambda: float(os.getenv("GRANITE_TEMPERATURE", "0.7")))
    top_p:          float = Field(default_factory=lambda: float(os.getenv("GRANITE_TOP_P", "0.9")))
    top_k:          int   = Field(default_factory=lambda: int(os.getenv("GRANITE_TOP_K", "50")))
    repetition_penalty: float = Field(default_factory=lambda: float(os.getenv("GRANITE_REP_PENALTY", "1.1")))


class VectorStoreConfig(BaseModel):
    persist_dir:     str = Field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", _DEFAULT_VECTOR_DIR))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    collection_name: str = "interview_kb"
    chunk_size:      int = 512
    chunk_overlap:   int = 64
    top_k_results:   int = 5


class KnowledgeBaseConfig(BaseModel):
    base_dir: str = Field(default_factory=lambda: os.getenv("KNOWLEDGE_BASE_DIR", _DEFAULT_KB_DIR))


class AppConfig(BaseModel):
    watsonx:        WatsonxConfig       = Field(default_factory=WatsonxConfig)
    vectorstore:    VectorStoreConfig   = Field(default_factory=VectorStoreConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    debug:          bool                = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    app_version:    str                 = "2.0.0"


# Singleton
config = AppConfig()
