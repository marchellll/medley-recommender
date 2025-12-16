"""SQLAlchemy models for the database."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Float, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Song(Base):
    """Song model storing metadata and file paths."""

    __tablename__ = "songs"

    song_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    youtube_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    lyrics: Mapped[str] = mapped_column(Text, nullable=False)

    # Audio file path
    audio_file_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Metadata
    bpm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    key: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Embedding file path
    embedding_file_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Song(song_id='{self.song_id}', title='{self.title}')>"




