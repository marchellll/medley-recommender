"""ANN index building using Chroma."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Song
from src.embeddings.encoder import load_embedding
from src.utils.config import settings


async def build_index(
    session: AsyncSession,
    embeddings_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> chromadb.Collection:
    """
    Build Chroma index from embedding files with metadata from database.

    Args:
        session: Database session to fetch song metadata
        embeddings_dir: Directory containing embedding JSON files
        output_dir: Directory to save Chroma database

    Returns:
        Chroma collection
    """
    if embeddings_dir is None:
        embeddings_dir = settings.embeddings_dir
    if output_dir is None:
        output_dir = settings.index_dir

    # Get all embedding files
    embedding_files = list(embeddings_dir.glob("*.json"))
    if not embedding_files:
        raise ValueError(f"No embedding files found in {embeddings_dir}")

    # Initialize Chroma client with persistence
    chroma_db_path = output_dir / "chroma_db"
    chroma_db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(chroma_db_path),
        )
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name="songs",
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )

    # Load embeddings and metadata
    embeddings = []
    ids = []
    metadatas = []

    # Fetch all songs from database to get metadata
    result = await session.execute(select(Song))
    songs_dict = {song.song_id: song for song in result.scalars().all()}

    for embedding_file in embedding_files:
        song_id = embedding_file.stem

        # Load embedding
        embedding = load_embedding(embedding_file)
        embeddings.append(embedding.tolist())

        # Get metadata from database
        song = songs_dict.get(song_id)
        metadata = {}
        if song:
            if song.key is not None:
                metadata["key"] = song.key
            if song.bpm is not None:
                metadata["bpm"] = float(song.bpm)

        ids.append(song_id)
        # Use empty dict if no metadata (Chroma requires dict, not None)
        metadatas.append(metadata if metadata else {})

    # Add to Chroma collection
    # Clear existing collection if rebuilding
    if collection.count() > 0:
        # Delete all existing items
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    # Add in batches for better performance
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]

        # All metadatas are already dicts (empty dict if no metadata)

        collection.add(
            embeddings=list(batch_embeddings),
            ids=list(batch_ids),
            metadatas=list(batch_metadatas),
        )

    # Persist to disk
    client.persist()

    return collection
