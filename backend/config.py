"""Application configuration using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """AquaQuery application settings loaded from environment variables."""

    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Data paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Path(__file__).parent.parent / "data"
    sample_data_dir: Path = Path(__file__).parent.parent / "data" / "sample"
    embeddings_dir: Path = Path(__file__).parent.parent / "data" / "embeddings"

    # argopy settings
    argo_cache_dir: str = str(Path(__file__).parent.parent / "data" / "sample")
    argo_fetch_timeout: int = 45  # seconds

    # ChromaDB settings
    chroma_collection_name: str = "argo_knowledge"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Rate limiting
    rate_limit_per_minute: int = 10

    # RAG settings
    rag_top_k: int = 3
    rag_min_relevance: float = 0.5

    model_config = {
        "env_prefix": "",
        "env_file": str(Path(__file__).parent.parent / ".env"),
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Create and return application settings."""
    return Settings()
