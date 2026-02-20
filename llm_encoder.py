"""BGE encoder for plot embeddings and user queries.

Uses BAAI/bge-large-en-v1.5 (1024-dim) for both plot encoding
and query encoding — must be the same model/space.
"""
import hashlib
from functools import lru_cache
from typing import Dict, List, Optional, Union

import torch

EMBEDDING_DIM = 1024
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


class LLMEncoder:
    """Encodes text into BGE embedding space.

    Same model is used for plots (documents) and queries (search).
    BGE recommends prefixing queries with an instruction for retrieval.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = None,
        cache_size: int = 10000,
    ):
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _add_to_cache(self, key: str, embedding: torch.Tensor):
        if key in self._cache:
            return
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = embedding
        self._cache_order.append(key)

    def encode(
        self,
        texts: Union[str, List[str]],
        return_tensors: bool = True,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Encode raw text(s) into BGE embeddings.

        For retrieval tasks, use encode_query or encode_plot instead.
        """
        if isinstance(texts, str):
            texts = [texts]

        results = [None] * len(texts)
        to_encode = []
        to_encode_idx = []

        if use_cache:
            for i, text in enumerate(texts):
                key = self._get_cache_key(text)
                if key in self._cache:
                    results[i] = self._cache[key]
                else:
                    to_encode.append(text)
                    to_encode_idx.append(i)
        else:
            to_encode = texts
            to_encode_idx = list(range(len(texts)))

        if to_encode:
            embeddings = self.model.encode(
                to_encode,
                convert_to_tensor=return_tensors,
                show_progress_bar=len(to_encode) > 100,
                normalize_embeddings=True,
            )
            if return_tensors:
                embeddings = embeddings.to(self._device)

            for idx, emb_idx in enumerate(to_encode_idx):
                if len(to_encode) == 1 and embeddings.dim() == 1:
                    emb = embeddings
                else:
                    emb = embeddings[idx]
                results[emb_idx] = emb
                if use_cache:
                    self._add_to_cache(self._get_cache_key(to_encode[idx]), emb)

        stacked = torch.stack(results)
        return stacked

    def encode_query(self, query: str) -> torch.Tensor:
        """Encode a user search query.

        BGE recommends prefixing queries with an instruction
        for asymmetric retrieval tasks.
        """
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.encode(prefixed, use_cache=False).squeeze(0)

    def encode_plot(self, plot_text: str) -> torch.Tensor:
        """Encode a movie plot/overview (document side)."""
        return self.encode(plot_text).squeeze(0)

    def encode_plots_batch(
        self, texts: List[str], batch_size: int = 64
    ) -> torch.Tensor:
        """Encode multiple plot texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = self.encode(batch, use_cache=False)
            all_embeddings.append(embs)
        return torch.cat(all_embeddings, dim=0)


class IntentParser:
    """Parse user messages for mood, genre, era, and constraints."""

    MOOD_KEYWORDS = {
        "cozy": ["cozy", "warm", "comfort", "heartwarming", "feel-good", "feelgood"],
        "exciting": ["exciting", "thrilling", "intense", "edge of seat", "adrenaline"],
        "scary": ["scary", "horror", "frightening", "terrifying", "creepy", "spooky"],
        "funny": ["funny", "comedy", "hilarious", "laugh", "humorous", "witty"],
        "sad": ["sad", "emotional", "cry", "tearjerker", "melancholy", "bittersweet"],
        "romantic": ["romantic", "love", "romance", "love story", "passionate"],
        "thoughtful": ["thoughtful", "deep", "philosophical", "thought-provoking", "cerebral", "mind-bending"],
        "relaxing": ["relaxing", "chill", "calm", "peaceful", "soothing", "easy-going"],
    }

    GENRE_KEYWORDS = {
        "action": ["action", "fight", "martial arts", "battle"],
        "comedy": ["comedy", "funny", "sitcom"],
        "drama": ["drama", "dramatic"],
        "horror": ["horror", "zombie", "slasher"],
        "scifi": ["sci-fi", "science fiction", "space", "futuristic", "cyberpunk"],
        "fantasy": ["fantasy", "magic", "wizard", "dragon"],
        "thriller": ["thriller", "suspense", "mystery", "detective"],
        "romance": ["romance", "romantic", "love story"],
        "documentary": ["documentary", "true story", "real life"],
        "animation": ["animated", "animation", "cartoon", "anime", "pixar"],
    }

    ERA_KEYWORDS = {
        "classic": ["classic", "old", "golden age", "vintage"],
        "80s": ["80s", "1980s", "eighties"],
        "90s": ["90s", "1990s", "nineties"],
        "2000s": ["2000s", "early 2000s"],
        "2010s": ["2010s", "twenty-tens"],
        "recent": ["recent", "new", "latest", "modern", "contemporary"],
    }

    def parse(self, message: str) -> dict:
        text = message.lower()
        return {
            "raw_text": message,
            "mood": self._extract_matches(text, self.MOOD_KEYWORDS),
            "genre": self._extract_matches(text, self.GENRE_KEYWORDS),
            "era": self._extract_matches(text, self.ERA_KEYWORDS),
            "constraints": self._extract_constraints(text),
        }

    def _extract_matches(self, text: str, keyword_dict: dict) -> list:
        matches = []
        for category, keywords in keyword_dict.items():
            if any(kw in text for kw in keywords):
                matches.append(category)
        return matches

    def _extract_constraints(self, text: str) -> dict:
        constraints = {}
        if any(w in text for w in ["short", "quick", "under 90", "90 min"]):
            constraints["max_duration"] = 90
        if any(w in text for w in ["long", "epic", "over 2 hours"]):
            constraints["min_duration"] = 120
        if any(w in text for w in ["family", "kids", "children", "family-friendly"]):
            constraints["family_friendly"] = True
        if any(w in text for w in ["mature", "adult", "r-rated", "graphic"]):
            constraints["mature_only"] = True
        return constraints
