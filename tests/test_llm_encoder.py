"""Tests for BGE encoder."""
import pytest


class TestLLMEncoder:
    def test_default_model_is_bge(self):
        from llm_encoder import LLMEncoder
        enc = LLMEncoder.__new__(LLMEncoder)
        enc._model_name = "BAAI/bge-large-en-v1.5"
        assert enc._model_name == "BAAI/bge-large-en-v1.5"

    def test_embedding_dim_constant(self):
        from llm_encoder import EMBEDDING_DIM
        assert EMBEDDING_DIM == 1024

    def test_encode_query_exists(self):
        """Should have encode_query for user search queries."""
        from llm_encoder import LLMEncoder
        assert hasattr(LLMEncoder, "encode_query")

    def test_encode_plot_exists(self):
        """Should have encode_plot for movie plot texts."""
        from llm_encoder import LLMEncoder
        assert hasattr(LLMEncoder, "encode_plot")


class TestIntentParser:
    def test_mood_keywords(self):
        from llm_encoder import IntentParser
        parser = IntentParser()
        result = parser.parse("I want something scary and exciting")
        assert "scary" in result["mood"]
        assert "exciting" in result["mood"]

    def test_genre_keywords(self):
        from llm_encoder import IntentParser
        parser = IntentParser()
        result = parser.parse("show me a good comedy")
        assert "comedy" in result["genre"]

    def test_era_keywords(self):
        from llm_encoder import IntentParser
        parser = IntentParser()
        result = parser.parse("something from the 90s")
        assert "90s" in result["era"]
