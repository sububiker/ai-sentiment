from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    model: str = "claude-opus-4-7"
    reflection_low: float = 0.40
    reflection_high: float = 0.60
    max_content_chars: int = 5000


settings = Settings()
