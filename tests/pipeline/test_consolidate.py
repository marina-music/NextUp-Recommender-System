"""Tests for Stage 4 consolidation."""

import polars as pl
from pipeline.consolidate import consolidate


class TestConsolidate:
    def test_combines_two_sources(self):
        dump_df = pl.DataFrame({
            "movieId": pl.Series([1, 2], dtype=pl.UInt32),
            "imdbId": ["tt0000001", "tt0000002"],
            "title": ["Movie A", "Movie B"],
            "year": pl.Series([2000, 2001], dtype=pl.UInt16),
            "plot_text": ["A long enough plot for movie A that has many words in it.", "Another plot for movie B with enough words."],
            "genres": ["Action|Drama", "Comedy"],
        })
        backfill_df = pl.DataFrame({
            "movieId": pl.Series([3], dtype=pl.UInt32),
            "imdbId": ["tt0000003"],
            "title": ["Movie C"],
            "year": pl.Series([2002], dtype=pl.UInt16),
            "plot_text": ["A backfilled plot for movie C with enough words."],
            "genres": ["Horror"],
        })
        result = consolidate(dump_df, backfill_df)
        assert len(result) == 3
        assert set(result["plot_source"].to_list()) == {"wiki_dump", "wikipedia_api"}
        assert "plot_length" in result.columns

    def test_deduplicates_on_movie_id(self):
        df1 = pl.DataFrame({
            "movieId": pl.Series([1], dtype=pl.UInt32),
            "imdbId": ["tt0000001"],
            "title": ["Movie A"],
            "year": pl.Series([2000], dtype=pl.UInt16),
            "plot_text": ["Dump version of the plot."],
            "genres": ["Drama"],
        })
        df2 = pl.DataFrame({
            "movieId": pl.Series([1], dtype=pl.UInt32),
            "imdbId": ["tt0000001"],
            "title": ["Movie A"],
            "year": pl.Series([2000], dtype=pl.UInt16),
            "plot_text": ["API version of the plot."],
            "genres": ["Drama"],
        })
        result = consolidate(df1, df2)
        assert len(result) == 1
        assert result["plot_source"][0] == "wiki_dump"
