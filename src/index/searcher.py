"""ANN index search functionality."""

from typing import Optional

import hnswlib
import numpy as np

from src.embeddings.encoder import generate_embedding
from src.index.builder import load_index
from src.utils.config import settings


class IndexSearcher:
    """Searcher for the ANN index."""

    def __init__(
        self,
        index: Optional[hnswlib.Index] = None,
        id_mapping: Optional[dict[int, str]] = None,
        dimension: Optional[int] = None,
    ):
        """
        Initialize searcher.

        Args:
            index: Optional pre-loaded index
            id_mapping: Optional pre-loaded ID mapping
            dimension: Embedding dimension (required if loading from disk)
        """
        self._index = index
        self._id_mapping = id_mapping
        self._dimension = dimension

    def load(self) -> None:
        """Load index and mapping from disk."""
        if self._dimension is None:
            # Try to detect dimension from first embedding file
            embedding_files = list(settings.embeddings_dir.glob("*.json"))
            if not embedding_files:
                raise ValueError("No embedding files found to determine dimension")
            from src.embeddings.encoder import load_embedding

            first_embedding = load_embedding(embedding_files[0])
            self._dimension = len(first_embedding)

        self._index, self._id_mapping = load_index(dimension=self._dimension)

    @property
    def index(self) -> hnswlib.Index:
        """Get index (loads if not already loaded)."""
        if self._index is None:
            self.load()
        return self._index

    @property
    def id_mapping(self) -> dict[int, str]:
        """Get ID mapping (loads if not already loaded)."""
        if self._id_mapping is None:
            self.load()
        return self._id_mapping

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Search for similar songs.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            ef_search: Optional ef_search parameter (overrides default)

        Returns:
            List of (song_id, distance) tuples, sorted by distance (ascending)
        """
        index = self.index
        id_mapping = self.id_mapping

        # Cap k to the number of elements in the index (can't return more than available)
        max_elements = index.element_count
        actual_k = min(k, max_elements)

        if actual_k == 0:
            return []

        # Set ef_search: must be at least k, preferably k * 2 for better recall
        if ef_search is None:
            # Ensure ef_search is at least k, with some margin for better recall
            # But also cap it reasonably for small datasets
            if settings.hnsw_auto_adjust:
                ef_search = min(
                    max(settings.hnsw_ef_search, actual_k * 2),
                    max_elements * 2  # Don't exceed 2x the dataset size
                )
            else:
                ef_search = max(settings.hnsw_ef_search, actual_k * 2)
        else:
            # Ensure provided ef_search is at least k
            ef_search = max(ef_search, actual_k)

        index.set_ef(ef_search)

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search with capped k
        labels, distances = index.knn_query(query_embedding, k=actual_k)

        # Convert to list of (song_id, distance) tuples
        results = []
        for label, distance in zip(labels[0], distances[0]):
            song_id = id_mapping.get(int(label))
            if song_id:
                results.append((song_id, float(distance)))

        return results

    def search_with_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_song_ids: Optional[set[str]],
        k: int = 10,
        max_k: int = 500,
        ef_search: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Search for similar songs, filtering to only include candidate song_ids.

        This is optimized for when you have pre-filtered candidates (e.g., by key/BPM).
        It will adaptively increase k until enough results are found or max_k is reached.

        Args:
            query_embedding: Query embedding vector
            candidate_song_ids: Set of song_ids to filter results to (None = no filtering)
            k: Target number of results to return
            max_k: Maximum k to search for (safety limit)
            ef_search: Optional ef_search parameter (overrides default)

        Returns:
            List of (song_id, distance) tuples, sorted by distance (ascending)
        """
        if candidate_song_ids is None or len(candidate_song_ids) == 0:
            # No filtering needed, use regular search
            return self.search(query_embedding, k=k, ef_search=ef_search)

        # Verify that at least some candidates exist in the index
        id_mapping = self.id_mapping
        # Create a set of all song_ids in the index for efficient lookup
        index_song_ids = set(id_mapping.values())
        candidates_in_index = candidate_song_ids & index_song_ids

        if not candidates_in_index:
            # No candidates found in index
            return []

        # Start with k, but increase adaptively if we don't get enough filtered results
        # Use a multiplier to account for filtering
        current_k = min(k * 3, max_k, len(candidates_in_index))
        results = []
        max_elements = self.index.element_count

        while len(results) < k and current_k <= max_elements and current_k <= max_k:
            # Search with current_k
            search_results = self.search(query_embedding, k=current_k, ef_search=ef_search)

            # Filter to only include candidates
            filtered = [
                (song_id, distance)
                for song_id, distance in search_results
                if song_id in candidate_song_ids
            ]

            results = filtered

            # If we got all available candidates or reached max_k, stop
            if len(results) >= len(candidates_in_index) or current_k >= max_k:
                break

            # Increase k for next iteration (exponential backoff)
            current_k = min(int(current_k * 1.5), max_k, max_elements)

        # Return top k results
        return results[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Search for similar songs by text query.

        Args:
            query_text: Query text (lyrics or description)
            k: Number of results to return
            ef_search: Optional ef_search parameter

        Returns:
            List of (song_id, distance) tuples, sorted by distance (ascending)
        """
        # Generate embedding for query text
        query_embedding = generate_embedding(query_text)

        return self.search(query_embedding, k=k, ef_search=ef_search)


