from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "PDF RAG Assistant"

    # embeddings
    embedding_backend: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "intfloat/e5-large-v2"
    embedding_batch_size: int = 64

    openai_api_key: str | None = None

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()  # type: ignore
