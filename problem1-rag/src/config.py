"""Runtime configuration, loaded from environment. No hardcoded secrets."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str
    llm_model: str
    judge_model: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    index_dir: str
    corpus_dir: str

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set - copy .env.example to .env")
        return cls(
            openrouter_api_key=key,
            openrouter_base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            llm_model=os.environ.get("LLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
            judge_model=os.environ.get("JUDGE_MODEL", "particle-ai/particle-1:free"),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", "384")),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "120")),
            top_k=int(os.environ.get("TOP_K", "5")),
            index_dir=os.environ.get("INDEX_DIR", "./data/index"),
            corpus_dir=os.environ.get("CORPUS_DIR", "./data/corpus"),
        )
