"""Tests for Wikidata movie/TV filtering."""
import polars as pl


class TestFilterPlots:
    def test_filter_to_movies_removes_non_movies(self):
        """Given raw plots and a wikidata result, filter correctly."""
        from pipeline.filter_plots import filter_plots_to_movies_tv

        raw = pl.DataFrame({
            "wiki_title": ["The Matrix", "Harry Potter (novel)", "Breaking Bad"],
            "plot_text": ["Neo discovers...", "Harry is a wizard...", "Walter White..."],
        })
        wikidata_titles = {"The Matrix", "Breaking Bad"}

        result = filter_plots_to_movies_tv(raw, wikidata_titles)
        assert len(result) == 2
        titles = result["wiki_title"].to_list()
        assert "The Matrix" in titles
        assert "Breaking Bad" in titles
        assert "Harry Potter (novel)" not in titles

    def test_filter_preserves_columns(self):
        """Output should have same columns as input."""
        from pipeline.filter_plots import filter_plots_to_movies_tv

        raw = pl.DataFrame({
            "wiki_title": ["The Matrix"],
            "plot_text": ["Neo discovers..."],
        })
        result = filter_plots_to_movies_tv(raw, {"The Matrix"})
        assert "wiki_title" in result.columns
        assert "plot_text" in result.columns
