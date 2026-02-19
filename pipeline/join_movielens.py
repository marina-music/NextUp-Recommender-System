"""Join WikiPlots data with MovieLens 32M via Wikidata bridge."""

import json
import re
from pathlib import Path

import polars as pl

from pipeline.download import DATA_DIR, REPORTS_DIR


def normalize_imdb_id(imdb_id: str) -> str:
    """Normalize an IMDb ID to 'tt' + 7-digit zero-padded format."""
    numeric = imdb_id.lstrip("t")
    return f"tt{numeric.zfill(7)}"


def parse_title_year(title: str) -> tuple[str, int | None]:
    """Parse MovieLens title format 'Movie Name (1995)' into (name, year)."""
    match = re.match(r"^(.+?)\s*\((\d{4})\)\s*$", title)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title.strip(), None


def join_with_movielens(
    wikiplots_with_imdb: pl.DataFrame,
    links_df: pl.DataFrame,
    movies_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join WikiPlots (with imdbId) against MovieLens links and movies.

    Args:
        wikiplots_with_imdb: DataFrame with wiki_title, imdbId, plot_text
        links_df: MovieLens links.csv as DataFrame (movieId, imdbId, tmdbId)
        movies_df: MovieLens movies.csv as DataFrame (movieId, title, genres)

    Returns:
        Joined DataFrame with movieId, imdbId, title, year, plot_text, genres
    """
    wikiplots_norm = wikiplots_with_imdb.with_columns(
        pl.col("imdbId").map_elements(normalize_imdb_id, return_dtype=pl.Utf8).alias("imdbId")
    )
    links_norm = links_df.with_columns(
        pl.col("imdbId").cast(pl.Utf8).map_elements(normalize_imdb_id, return_dtype=pl.Utf8).alias("imdbId")
    )

    joined = wikiplots_norm.join(links_norm.select(["movieId", "imdbId"]), on="imdbId", how="inner")

    movies_parsed = movies_df.with_columns([
        pl.col("title").map_elements(lambda t: parse_title_year(t)[0], return_dtype=pl.Utf8).alias("parsed_title"),
        pl.col("title").map_elements(lambda t: parse_title_year(t)[1], return_dtype=pl.UInt16).alias("year"),
    ])

    result = joined.join(
        movies_parsed.select(["movieId", "parsed_title", "year", "genres"]),
        on="movieId",
        how="left",
    ).rename({"parsed_title": "title"}).select([
        "movieId", "imdbId", "title", "year", "plot_text", "genres",
    ])

    return result


def run_stage2(
    wiki_plots_raw: pl.DataFrame,
    wikidata_mapping: pl.DataFrame,
    movielens_dir: Path,
) -> pl.DataFrame:
    """Execute Stage 2: bridge WikiPlots to MovieLens.

    Args:
        wiki_plots_raw: Output of Stage 1 (wiki_title, plot_text)
        wikidata_mapping: Wikidata bridge (wiki_title, imdbId)
        movielens_dir: Path to extracted MovieLens 32M directory

    Returns:
        Matched DataFrame ready for Stage 4
    """
    wikiplots_with_imdb = wiki_plots_raw.join(wikidata_mapping, on="wiki_title", how="inner")
    links_df = pl.read_csv(movielens_dir / "links.csv")
    movies_df = pl.read_csv(movielens_dir / "movies.csv")
    matched = join_with_movielens(wikiplots_with_imdb, links_df, movies_df)

    wikidata_matched = len(wikiplots_with_imdb)
    movielens_matched = len(matched)
    total_wikiplots = len(wiki_plots_raw)
    total_movielens = len(links_df)

    report = {
        "total_wikiplots_extracted": total_wikiplots,
        "wikiplots_with_wikidata_match": wikidata_matched,
        "wikidata_match_rate": f"{wikidata_matched / max(total_wikiplots, 1) * 100:.1f}%",
        "total_movielens_movies": total_movielens,
        "wikiplots_in_movielens": movielens_matched,
        "movielens_match_rate": f"{movielens_matched / max(total_movielens, 1) * 100:.1f}%",
    }
    report_path = REPORTS_DIR / "stage2_bridge_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Stage 2 report: {report}")

    unmatched_wikiplots = wiki_plots_raw.join(wikidata_mapping, on="wiki_title", how="anti")
    unmatched_path = REPORTS_DIR / "stage2_unmatched_wikiplots.parquet"
    unmatched_wikiplots.write_parquet(unmatched_path)

    matched_path = REPORTS_DIR / "stage2_matched_movies.parquet"
    matched.write_parquet(matched_path)

    return matched
