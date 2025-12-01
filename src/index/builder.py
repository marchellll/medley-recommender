"""ANN index building using HNSWLib."""

import json
from pathlib import Path
from typing import Optional

import hnswlib
import numpy as np

from src.embeddings.encoder import load_embedding
from src.utils.config import settings


def build_index(
    embeddings_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dimension: Optional[int] = None,
) -> tuple[hnswlib.Index, dict[str, int]]:
    """
    Build HNSWLib index from embedding files.

    Args:
        embeddings_dir: Directory containing embedding JSON files
        output_dir: Directory to save index files
        dimension: Embedding dimension (auto-detected if None)

    Returns:
        Tuple of (index, id_mapping) where id_mapping maps index IDs to song IDs
    """
    if embeddings_dir is None:
        embeddings_dir = settings.embeddings_dir
    if output_dir is None:
        output_dir = settings.index_dir

    # Get all embedding files
    embedding_files = list(embeddings_dir.glob("*.json"))
    if not embedding_files:
        raise ValueError(f"No embedding files found in {embeddings_dir}")

    # Load first embedding to get dimension
    if dimension is None:
        first_embedding = load_embedding(embedding_files[0])
        dimension = len(first_embedding)
    else:
        first_embedding = load_embedding(embedding_files[0])
        if len(first_embedding) != dimension:
            raise ValueError(
                f"Dimension mismatch: expected {dimension}, got {len(first_embedding)}"
            )

    # Load all embeddings
    embeddings = []
    song_ids = []
    id_mapping = {}  # Maps index ID to song ID

    for idx, embedding_file in enumerate(embedding_files):
        song_id = embedding_file.stem
        embedding = load_embedding(embedding_file)

        if len(embedding) != dimension:
            raise ValueError(
                f"Dimension mismatch for {song_id}: expected {dimension}, got {len(embedding)}"
            )

        embeddings.append(embedding)
        song_ids.append(song_id)
        id_mapping[idx] = song_id

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Adjust parameters for small datasets if auto-adjust is enabled
    num_elements = len(embeddings)
    if settings.hnsw_auto_adjust:
        # M should be at least 2, but for very small datasets, cap it
        # M cannot exceed num_elements - 1 (each node needs at least one other node to connect to)
        adjusted_m = min(settings.hnsw_m, max(2, num_elements - 1))
        # ef_construction should be reasonable for small datasets
        # At minimum: M * 2, but cap it to avoid excessive computation
        adjusted_ef_construction = min(
            settings.hnsw_ef_construction,
            max(adjusted_m * 2, num_elements * 2, 10)
        )
    else:
        adjusted_m = settings.hnsw_m
        adjusted_ef_construction = settings.hnsw_ef_construction

    # Create index
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.init_index(
        max_elements=num_elements,
        ef_construction=adjusted_ef_construction,
        M=adjusted_m,
    )

    # Add embeddings to index
    index.add_items(embeddings_array, list(range(len(embeddings))))

    # Set ef_search for query time
    index.set_ef(settings.hnsw_ef_search)

    # Save index
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "hnsw_index.bin"
    index.save_index(str(index_path))

    # Save ID mapping
    mapping_path = output_dir / "id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)

    return index, id_mapping


def load_index(
    index_dir: Optional[Path] = None,
    dimension: Optional[int] = None,
) -> tuple[hnswlib.Index, dict[str, int]]:
    """
    Load HNSWLib index from disk.

    Args:
        index_dir: Directory containing index files
        dimension: Embedding dimension (required for loading)

    Returns:
        Tuple of (index, id_mapping)
    """
    if index_dir is None:
        index_dir = settings.index_dir
    if dimension is None:
        raise ValueError("Dimension must be provided to load index")

    index_path = index_dir / "hnsw_index.bin"
    mapping_path = index_dir / "id_mapping.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    # Load index
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.load_index(str(index_path))
    index.set_ef(settings.hnsw_ef_search)

    # Load ID mapping
    with open(mapping_path, "r") as f:
        id_mapping = json.load(f)
        # Convert string keys to int
        id_mapping = {int(k): v for k, v in id_mapping.items()}

    return index, id_mapping


