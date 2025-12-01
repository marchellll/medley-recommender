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

    # Index Configuration
    # HNSW_M: Number of bi-directional links per node (higher = better recall, more memory)
    # Recommended: 16 for >1000 songs, 8 for 100-1000 songs, 4 for 10-100 songs, 2 for <10 songs
    hnsw_m: int = Field(default=16, alias="HNSW_M")

    # HNSW_EF_CONSTRUCTION: Search width during construction (higher = better quality, slower build)
    # Recommended: 200 for >1000 songs, 100 for 100-1000 songs, 50 for 10-100 songs, 10 for <10 songs
    hnsw_ef_construction: int = Field(default=200, alias="HNSW_EF_CONSTRUCTION")

    # HNSW_EF_SEARCH: Search width during query (higher = better recall, slower search)
    # Recommended: 50 for >1000 songs, 30 for 100-1000 songs, 20 for 10-100 songs, 10 for <10 songs
    hnsw_ef_search: int = Field(default=50, alias="HNSW_EF_SEARCH")

    # HNSW_AUTO_ADJUST: Automatically adjust parameters for small datasets (recommended: true)
    hnsw_auto_adjust: bool = Field(default=True, alias="HNSW_AUTO_ADJUST")

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


