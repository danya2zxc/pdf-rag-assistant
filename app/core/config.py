from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PDF RAG Assistant"
    openai_api_key: str | None = None

    
    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()
