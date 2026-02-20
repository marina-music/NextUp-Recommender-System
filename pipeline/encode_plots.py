"""Encode movie plots with BGE and build FAISS index."""
import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import polars as pl
from tqdm import tqdm

from pipeline.download import DATA_DIR, REPORTS_DIR

EMBEDDING_DIM = 1024
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def prepend_metadata(
    plot_text: str,
    media_type: str = "Film",
    genres: Optional[str] = None,
    year: Optional[int] = None,
    title: Optional[str] = None,
) -> str:
    """Prepend structured metadata to plot text for embedding normalization."""
    parts = [media_type]
    if genres:
        parts.append(genres)
    if year is not None:
        parts.append(str(year))
    if title:
        parts.append(title)
    prefix = ". ".join(parts)
    return f"{prefix}. {plot_text}"


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from embeddings.

    Note: This function does NOT normalize the input embeddings.
    Callers are responsible for providing already-normalized vectors
    if cosine similarity via inner product is desired.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def encode_and_build_index(
    plots_df: pl.DataFrame,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    index_path: Path = DATA_DIR / "plots.faiss",
    metadata_path: Path = DATA_DIR / "plots_metadata.parquet",
):
    """Encode all plots and build FAISS index.

    Args:
        plots_df: DataFrame with columns: wiki_title or title, plot_text,
                  and optionally genres, year, plot_source.
        model_name: Sentence transformer model name.
        batch_size: Encoding batch size.
        index_path: Where to save the FAISS index.
        metadata_path: Where to save the metadata mapping.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Determine title column
    title_col = "title" if "title" in plots_df.columns else "wiki_title"

    # Prepare texts with metadata
    texts = []
    for row in tqdm(plots_df.iter_rows(named=True), total=len(plots_df), desc="Preparing texts"):
        title = row.get(title_col, "")
        genres = row.get("genres", None)
        year = row.get("year", None)
        media_type = "Film"
        if genres and "Animation" in str(genres):
            media_type = "Animated Film"

        text = prepend_metadata(row["plot_text"], media_type, genres, year, title)
        texts.append(text)

    # Encode in batches
    print(f"Encoding {len(texts)} plots with {model_name}...")
    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    all_embeddings = all_embeddings.astype(np.float32)
    print(f"Embeddings shape: {all_embeddings.shape}")

    # Build FAISS index
    index = faiss.IndexFlatIP(all_embeddings.shape[1])
    index.add(all_embeddings)
    print(f"FAISS index: {index.ntotal} vectors, {index.d} dimensions")

    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    # Save metadata (position -> movie info mapping)
    metadata_cols = [c for c in plots_df.columns if c != "plot_text"]
    metadata_df = plots_df.select(metadata_cols).with_row_index("faiss_idx")
    metadata_df.write_parquet(metadata_path)
    print(f"Saved metadata to {metadata_path}")

    # Report
    report = {
        "total_encoded": len(texts),
        "embedding_dim": int(all_embeddings.shape[1]),
        "model": model_name,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }
    report_path = REPORTS_DIR / "encode_plots_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report}")


def main():
    """Encode plots and build index from movie_plots.parquet."""
    plots_path = DATA_DIR / "movie_plots.parquet"
    if not plots_path.exists():
        print(f"ERROR: {plots_path} not found.")
        return

    plots_df = pl.read_parquet(plots_path)
    print(f"Loaded {len(plots_df)} plots")

    encode_and_build_index(plots_df)


if __name__ == "__main__":
    main()
