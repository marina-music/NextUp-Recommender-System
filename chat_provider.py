"""Provider-agnostic chatbot LLM for formatting recommendations."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


def format_prompt(query: str, recommendations: List[Dict], max_recs: int = 5) -> str:
    """Format the recommendation prompt for any LLM provider."""
    recs_text = ""
    for i, rec in enumerate(recommendations[:max_recs], 1):
        title = rec.get("title", "Unknown")
        year = rec.get("year", "")
        genres = rec.get("genres", "")
        snippet = rec.get("plot_snippet", "")
        line = f"{i}. {title}"
        if year:
            line += f" ({year})"
        if genres:
            line += f" - {genres}"
        if snippet:
            line += f" - {snippet}"
        recs_text += line + "\n"

    return (
        f'The user asked: "{query}"\n'
        f"Based on their taste profile and history, here are the top recommendations:\n"
        f"{recs_text}\n"
        f"Generate a natural, conversational response presenting these recommendations. "
        f"Explain briefly why each might appeal to the user based on their query."
    )


class ChatProvider(ABC):
    """Abstract base class for chat LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        ...


class OpenAIChat(ChatProvider):
    """OpenAI GPT chat provider."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key

    def generate(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a friendly movie recommendation assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content


class ClaudeChat(ChatProvider):
    """Anthropic Claude chat provider."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key

    def generate(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=500,
            system="You are a friendly movie recommendation assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class GeminiChat(ChatProvider):
    """Google Gemini chat provider."""

    def __init__(self, model: str = "gemini-pro", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key

    def generate(self, prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model)
        response = model.generate_content(
            f"You are a friendly movie recommendation assistant.\n\n{prompt}"
        )
        return response.text


def create_provider(provider: str, model: Optional[str] = None, api_key: Optional[str] = None) -> ChatProvider:
    """Factory function to create a chat provider."""
    providers = {
        "openai": OpenAIChat,
        "claude": ClaudeChat,
        "gemini": GeminiChat,
    }
    cls = providers.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    kwargs = {}
    if model:
        kwargs["model"] = model
    if api_key:
        kwargs["api_key"] = api_key
    return cls(**kwargs)
