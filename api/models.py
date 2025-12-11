"""Pydantic models for API requests and responses."""

from typing import Optional

from pydantic import BaseModel, HttpUrl


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    keys: Optional[list[str]] = None
    limit: int = 10


class AddSongRequest(BaseModel):
    """Add song request model."""

    title: str
    youtube_url: HttpUrl
    lyrics: str


class SongResult(BaseModel):
    """Song result model."""

    song_id: str
    title: str
    bpm: Optional[float] = None
    key: Optional[str] = None
    similarity_score: float
    youtube_url: str


class SearchResponse(BaseModel):
    """Search response model."""

    results: list[SongResult]
    total: int


class AddSongResponse(BaseModel):
    """Add song response model."""

    success: bool
    song_id: Optional[str] = None
    message: str


