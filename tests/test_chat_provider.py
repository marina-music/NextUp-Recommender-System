"""Tests for chat provider interface."""
import pytest


class TestChatProvider:
    def test_abstract_interface(self):
        from chat_provider import ChatProvider
        with pytest.raises(TypeError):
            ChatProvider()  # abstract, can't instantiate

    def test_format_recommendations_prompt(self):
        from chat_provider import format_prompt
        recs = [
            {"title": "The Matrix", "year": 1999, "genres": "Action, Sci-Fi", "plot_snippet": "A hacker discovers..."},
            {"title": "Inception", "year": 2010, "genres": "Sci-Fi, Thriller", "plot_snippet": "A thief enters..."},
        ]
        prompt = format_prompt("something mind-bending", recs)
        assert "something mind-bending" in prompt
        assert "The Matrix" in prompt
        assert "Inception" in prompt

    def test_openai_provider_exists(self):
        from chat_provider import OpenAIChat
        assert hasattr(OpenAIChat, "generate")

    def test_claude_provider_exists(self):
        from chat_provider import ClaudeChat
        assert hasattr(ClaudeChat, "generate")
