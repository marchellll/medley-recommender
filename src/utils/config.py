"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_path: str = Field(default="data/medley.db", alias="DATABASE_PATH")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=9876, alias="API_PORT")

    # Streamlit Configuration
    streamlit_port: int = Field(default=9877, alias="STREAMLIT_PORT")

    # Embedding Model
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def database_path_resolved(self) -> Path:
        """Get resolved database path."""
        return Path(self.database_path).resolve()

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path("data").resolve()

    @property
    def audio_dir(self) -> Path:
        """Get audio directory path."""
        return self.data_dir / "audio"

    @property
    def embeddings_dir(self) -> Path:
        """Get embeddings directory path."""
        return self.data_dir / "songs_with_embeddings"

    @property
    def index_dir(self) -> Path:
        """Get index directory path."""
        return self.data_dir / "index"

    @property
    def new_songs_dir(self) -> Path:
        """Get new songs directory path."""
        return self.data_dir / "new_songs"


# Global settings instance
settings = Settings()


