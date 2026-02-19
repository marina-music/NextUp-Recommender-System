"""Stage 4: Consolidate matched and backfilled plots into final Parquet."""

import json
import re
from pathlib import Path

import polars as pl

from pipeline.download import DATA_DIR, REPORTS_DIR


def consolidate(
    dump_matched: pl.DataFrame,
    backfill_df: pl.DataFrame,
) -> pl.DataFrame:
    """Combine dump-matched and API-backfilled plots.

    Deduplicates on movieId, preferring dump_matched (wiki_dump source).
    """
    dump_tagged = dump_matched.with_columns(pl.lit("wiki_dump").alias("plot_source"))
    backfill_tagged = backfill_df.with_columns(pl.lit("wikipedia_api").alias("plot_source"))

    common_cols = ["movieId", "imdbId", "title", "year", "plot_text", "genres", "plot_source"]
    dump_tagged = dump_tagged.select([c for c in common_cols if c in dump_tagged.columns])
    backfill_tagged = backfill_tagged.select([c for c in common_cols if c in backfill_tagged.columns])

    combined = pl.concat([dump_tagged, backfill_tagged], how="diagonal_relaxed")

    combined = combined.sort("plot_source")
    combined = combined.unique(subset=["movieId"], keep="first")

    combined = combined.with_columns(
        pl.col("plot_text").str.split(" ").list.len().alias("plot_length").cast(pl.UInt32)
    )

    return combined.sort("movieId")


def write_final_reports(
    final_df: pl.DataFrame,
    links_df: pl.DataFrame,
    movies_df: pl.DataFrame,
):
    """Write Stage 4 quality reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    total_movielens = len(links_df)
    total_with_plots = len(final_df)
    from_dump = final_df.filter(pl.col("plot_source") == "wiki_dump").height
    from_api = final_df.filter(pl.col("plot_source") == "wikipedia_api").height
    missing = total_movielens - total_with_plots

    lengths = final_df["plot_length"]
    length_stats = {
        "min": int(lengths.min()) if lengths.len() > 0 else 0,
        "p25": int(lengths.quantile(0.25)) if lengths.len() > 0 else 0,
        "median": int(lengths.median()) if lengths.len() > 0 else 0,
        "p75": int(lengths.quantile(0.75)) if lengths.len() > 0 else 0,
        "max": int(lengths.max()) if lengths.len() > 0 else 0,
    }

    report = {
        "total_movielens_movies": total_movielens,
        "movies_with_plot": total_with_plots,
        "coverage_pct": f"{total_with_plots / max(total_movielens, 1) * 100:.1f}%",
        "from_wiki_dump": from_dump,
        "from_wikipedia_api": from_api,
        "still_missing": missing,
        "missing_pct": f"{missing / max(total_movielens, 1) * 100:.1f}%",
        "plot_length_distribution": length_stats,
    }

    if "genres" in final_df.columns and "genres" in movies_df.columns:
        all_genres = set()
        for genre_str in movies_df["genres"].drop_nulls().to_list():
            all_genres.update(genre_str.split("|"))
        genre_rates = {}
        for genre in sorted(all_genres - {"(no genres listed)"}):
            total_genre = movies_df.filter(pl.col("genres").str.contains(genre)).height
            matched_genre = final_df.filter(pl.col("genres").str.contains(genre)).height
            if total_genre > 0:
                genre_rates[genre] = f"{matched_genre}/{total_genre} ({matched_genre / total_genre * 100:.0f}%)"
        report["per_genre_match_rate"] = genre_rates

    report_path = REPORTS_DIR / "stage4_final_report.json"
    with open(report_path, "w") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"Stage 4 report: {json.dumps(report, indent=2)}")

    matched_ids = set(final_df["imdbId"].to_list())

    def _parse_year(title):
        match = re.match(r"^.+?\((\d{4})\)\s*$", title)
        return int(match.group(1)) if match else None

    def _parse_title(title):
        match = re.match(r"^(.+?)\s*\(\d{4}\)\s*$", title)
        return match.group(1).strip() if match else title.strip()

    no_plot = (
        links_df.join(movies_df, on="movieId", how="left")
        .with_columns(pl.col("imdbId").cast(pl.Utf8))
        .filter(~pl.col("imdbId").is_in(list(matched_ids)))
        .with_columns([
            pl.col("title").map_elements(_parse_title, return_dtype=pl.Utf8).alias("clean_title"),
            pl.col("title").map_elements(_parse_year, return_dtype=pl.UInt16).alias("year"),
        ])
        .select(["imdbId", "clean_title", "year", "genres"])
        .rename({"clean_title": "title"})
        .sort("year", descending=True)
    )

    no_plot.write_parquet(REPORTS_DIR / "stage4_no_plot_movies.parquet")
    no_plot.write_csv(REPORTS_DIR / "stage4_no_plot_movies.csv")
    print(f"  Wrote {len(no_plot)} unmatched movies to stage4_no_plot_movies.csv")
