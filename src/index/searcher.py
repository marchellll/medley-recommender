"""ANN index search functionality using Chroma."""

from typing import Optional

import chromadb
import numpy as np

from src.embeddings.encoder import generate_embedding
from src.utils.config import settings


class IndexSearcher:
    """Searcher for the Chroma index."""

    def __init__(
        self,
        collection: Optional[chromadb.Collection] = None,
    ):
        """
        Initialize searcher.

        Args:
            collection: Optional pre-loaded Chroma collection
        """
        self._collection = collection

    def load(self) -> None:
        """Load Chroma collection from disk."""
        chroma_db_path = settings.index_dir / "chroma_db"

        if not chroma_db_path.exists():
            raise FileNotFoundError(
                f"Chroma database not found: {chroma_db_path}. Please run the pipeline to build the index."
            )

        client = chromadb.PersistentClient(path=str(chroma_db_path))

        self._collection = client.get_collection(name="songs")

    @property
    def collection(self) -> chromadb.Collection:
        """Get collection (loads if not already loaded)."""
        if self._collection is None:
            self.load()
        return self._collection

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        keys: Optional[list[str]] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """
        Search for similar songs with optional filtering.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            keys: Optional list of keys to filter by
            bpm_min: Optional minimum BPM filter
            bpm_max: Optional maximum BPM filter

        Returns:
            List of (song_id, distance) tuples, sorted by distance (ascending)
        """
        collection = self.collection

        # Build filter dict for Chroma
        where = {}
        if keys is not None and len(keys) > 0:
            where["key"] = {"$in": keys}
        if bpm_min is not None or bpm_max is not None:
            bpm_filter = {}
            if bpm_min is not None:
                bpm_filter["$gte"] = bpm_min
            if bpm_max is not None:
                bpm_filter["$lte"] = bpm_max
            if bpm_filter:
                where["bpm"] = bpm_filter

        # Convert numpy array to list for Chroma
        # Chroma expects query_embeddings as a list of lists (even for single query)
        query_embedding_list = query_embedding.tolist()

        # Ensure it's a list of lists (single query)
        if not isinstance(query_embedding_list[0], list):
            query_embedding_list = [query_embedding_list]

        # Query Chroma collection
        results = collection.query(
            query_embeddings=query_embedding_list,
            n_results=k,
            where=where if where else None,
        )

        # Convert results to list of (song_id, distance) tuples
        if not results["ids"] or len(results["ids"][0]) == 0:
            return []

        song_ids = results["ids"][0]
        distances = results["distances"][0]

        return [(song_id, float(distance)) for song_id, distance in zip(song_ids, distances)]

    def search_by_text(
        self,
        query_text: str,
        k: int = 10,
        keys: Optional[list[str]] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """
        Search for similar songs by text query.

        Args:
            query_text: Query text (lyrics or description)
            k: Number of results to return
            keys: Optional list of keys to filter by
            bpm_min: Optional minimum BPM filter
            bpm_max: Optional maximum BPM filter

        Returns:
            List of (song_id, distance) tuples, sorted by distance (ascending)
        """
        # Generate embedding for query text
        query_embedding = generate_embedding(query_text)

        return self.search(
            query_embedding,
            k=k,
            keys=keys,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
        )
