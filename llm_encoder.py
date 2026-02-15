"""
LLM Encoder Module for Mamba4Rec Fusion

This module provides utilities for encoding user messages and movie descriptions
into embeddings using sentence transformers.

The encoder:
- Uses a pre-trained sentence transformer model
- Supports batched encoding for efficiency
- Provides intent parsing to extract structured mood information
- Caches embeddings for repeated queries
"""

import torch
from typing import List, Dict, Optional, Union
from functools import lru_cache
import hashlib


class LLMEncoder:
    """
    Encoder for converting text to embeddings using sentence transformers.

    Args:
        model_name: Name of the sentence transformer model
                   (default: "all-MiniLM-L6-v2" for speed, or
                    "all-mpnet-base-v2" for quality)
        device: Device to run the model on
        cache_size: Number of embeddings to cache
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_size: int = 10000
    ):
        self.model_name = model_name
        self.device = device
        self.cache_size = cache_size

        # Lazy loading of sentence transformers
        self._model = None
        self._embedding_dim = None

        # Simple cache using LRU
        self._cache = {}
        self._cache_order = []

    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        if self._embedding_dim is None:
            _ = self.model  # Trigger lazy loading
        return self._embedding_dim

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _add_to_cache(self, key: str, embedding: torch.Tensor):
        """Add embedding to cache with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # Evict oldest entry
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = embedding
        self._cache_order.append(key)

    def encode(
        self,
        texts: Union[str, List[str]],
        return_tensors: bool = True,
        use_cache: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text or list of texts to encode
            return_tensors: If True, return PyTorch tensors
            use_cache: If True, use caching for repeated queries

        Returns:
            Embeddings as tensor [B, embedding_dim] or list of tensors
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings = []
        texts_to_encode = []
        text_indices = []

        # Check cache
        for i, text in enumerate(texts):
            if use_cache:
                key = self._get_cache_key(text)
                if key in self._cache:
                    embeddings.append((i, self._cache[key]))
                    continue

            texts_to_encode.append(text)
            text_indices.append(i)

        # Encode uncached texts
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                convert_to_tensor=True,
                device=self.device
            )

            for idx, (i, text) in enumerate(zip(text_indices, texts_to_encode)):
                emb = new_embeddings[idx]
                if use_cache:
                    key = self._get_cache_key(text)
                    self._add_to_cache(key, emb)
                embeddings.append((i, emb))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        embeddings = [e[1] for e in embeddings]

        if return_tensors:
            embeddings = torch.stack(embeddings)

        if single_input:
            return embeddings[0] if return_tensors else embeddings[0]

        return embeddings

    def encode_mood(self, user_message: str) -> torch.Tensor:
        """
        Encode a user mood/intent message.

        This is a convenience method that applies any mood-specific preprocessing.

        Args:
            user_message: User's mood description (e.g., "I want a cozy movie")

        Returns:
            Mood embedding [embedding_dim]
        """
        # Optionally preprocess the message to emphasize mood keywords
        processed = self._preprocess_mood(user_message)
        return self.encode(processed, return_tensors=True)

    def _preprocess_mood(self, message: str) -> str:
        """
        Preprocess mood message to emphasize relevant terms.

        This can be customized based on your application needs.
        """
        # For now, just clean and normalize
        message = message.strip().lower()

        # Add context for better embedding
        if not any(word in message for word in ["movie", "film", "watch"]):
            message = f"I want to watch a movie that is {message}"

        return message

    def encode_movie_description(self, description: str) -> torch.Tensor:
        """
        Encode a movie description for alignment training.

        Args:
            description: Movie description text

        Returns:
            Description embedding [embedding_dim]
        """
        return self.encode(description, return_tensors=True)


class IntentParser:
    """
    Parser for extracting structured intent from user messages.

    Extracts:
    - mood: emotional tone (cozy, exciting, scary, etc.)
    - genre: movie genre preferences
    - era: time period preferences
    - style: directorial/cinematographic style
    - constraints: specific requirements (not too long, no violence, etc.)
    """

    # Mood keywords
    MOOD_KEYWORDS = {
        "cozy": ["cozy", "warm", "comfortable", "comforting", "heartwarming"],
        "exciting": ["exciting", "thrilling", "action", "intense", "adrenaline"],
        "scary": ["scary", "horror", "frightening", "creepy", "spooky"],
        "funny": ["funny", "comedy", "hilarious", "laughing", "humorous"],
        "sad": ["sad", "depressing", "emotional", "crying", "tearjerker"],
        "romantic": ["romantic", "love", "romance", "relationship"],
        "thoughtful": ["thoughtful", "deep", "philosophical", "thought-provoking"],
        "relaxing": ["relaxing", "chill", "calm", "peaceful", "laid-back"],
    }

    # Genre keywords
    GENRE_KEYWORDS = {
        "action": ["action", "fighting", "combat", "martial arts"],
        "comedy": ["comedy", "funny", "humor", "sitcom"],
        "drama": ["drama", "dramatic", "serious"],
        "horror": ["horror", "scary", "slasher", "supernatural"],
        "scifi": ["sci-fi", "science fiction", "space", "futuristic", "cyberpunk"],
        "fantasy": ["fantasy", "magical", "dragons", "wizards"],
        "thriller": ["thriller", "suspense", "mystery", "crime"],
        "romance": ["romance", "romantic", "love story"],
        "documentary": ["documentary", "true story", "real life"],
        "animation": ["animation", "animated", "cartoon", "anime"],
    }

    # Era keywords
    ERA_KEYWORDS = {
        "classic": ["classic", "old", "vintage", "golden age"],
        "80s": ["80s", "1980s", "eighties"],
        "90s": ["90s", "1990s", "nineties"],
        "2000s": ["2000s", "early 2000s", "aughts"],
        "2010s": ["2010s", "twenty tens"],
        "recent": ["recent", "new", "latest", "modern", "contemporary"],
    }

    def parse(self, message: str) -> Dict[str, any]:
        """
        Parse user message to extract structured intent.

        Args:
            message: User's movie preference message

        Returns:
            Dictionary with extracted intent components
        """
        message_lower = message.lower()

        intent = {
            "raw_text": message,
            "mood": self._extract_matches(message_lower, self.MOOD_KEYWORDS),
            "genre": self._extract_matches(message_lower, self.GENRE_KEYWORDS),
            "era": self._extract_matches(message_lower, self.ERA_KEYWORDS),
            "constraints": self._extract_constraints(message_lower),
        }

        return intent

    def _extract_matches(
        self,
        text: str,
        keyword_dict: Dict[str, List[str]]
    ) -> List[str]:
        """Extract matching categories from text."""
        matches = []
        for category, keywords in keyword_dict.items():
            if any(kw in text for kw in keywords):
                matches.append(category)
        return matches

    def _extract_constraints(self, text: str) -> Dict[str, any]:
        """Extract specific constraints from text."""
        constraints = {}

        # Length constraints
        if any(phrase in text for phrase in ["not too long", "short", "under 2 hours"]):
            constraints["max_duration"] = 120
        if any(phrase in text for phrase in ["long", "epic", "over 2 hours"]):
            constraints["min_duration"] = 120

        # Content constraints
        if any(phrase in text for phrase in ["no violence", "family friendly", "kid friendly"]):
            constraints["family_friendly"] = True
        if any(phrase in text for phrase in ["adult", "mature", "not for kids"]):
            constraints["mature_only"] = True

        return constraints


# Convenience function for quick encoding
def encode_user_mood(
    message: str,
    encoder: Optional[LLMEncoder] = None
) -> torch.Tensor:
    """
    Quick function to encode a user mood message.

    Args:
        message: User's mood/intent message
        encoder: Optional pre-initialized encoder

    Returns:
        Mood embedding tensor
    """
    if encoder is None:
        encoder = LLMEncoder()
    return encoder.encode_mood(message)
