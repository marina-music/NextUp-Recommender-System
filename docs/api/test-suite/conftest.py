"""Shared fixtures for CineMatch External Recommender contract tests.

Configure via environment variables:
    RECOMMENDER_URL  — Base URL (default: http://localhost:9000)
    RECOMMENDER_USER — Basic Auth username (default: test)
    RECOMMENDER_PASS — Basic Auth password (default: test)
"""

import os
from uuid import uuid4

import httpx
import pytest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("RECOMMENDER_URL", "http://localhost:9000")
AUTH_USER = os.environ.get("RECOMMENDER_USER", "test")
AUTH_PASS = os.environ.get("RECOMMENDER_PASS", "test")


# ---------------------------------------------------------------------------
# HTTP client fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """Authenticated httpx client pointed at the recommender service."""
    with httpx.Client(
        base_url=BASE_URL,
        auth=(AUTH_USER, AUTH_PASS),
        timeout=15.0,
    ) as c:
        yield c


@pytest.fixture
def unauthed_client():
    """Client with no authentication — for testing 401 responses."""
    with httpx.Client(base_url=BASE_URL, timeout=15.0) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------
@pytest.fixture
def new_user_request():
    """Minimal request: new user with genre preferences but no history."""
    return {
        "user_id": str(uuid4()),
        "query": None,
        "limit": 20,
        "region": "US",
        "profile": {
            "preferred_genres": ["Action", "Sci-Fi"],
            "preferred_services": ["netflix", "prime_video"],
            "disliked_genres": [],
            "description": None,
            "traits": [],
            "recommendation_focus": {},
        },
        "interaction_history": {
            "recent_events": [],
            "watched": [],
            "watchlist": [],
            "exclude": [],
        },
    }


@pytest.fixture
def power_user_request():
    """Rich request: detailed profile + interaction history + exclusion set."""
    return {
        "user_id": str(uuid4()),
        "query": "atmospheric horror with great cinematography",
        "limit": 15,
        "region": "US",
        "profile": {
            "preferred_genres": ["Horror", "Thriller", "Mystery"],
            "preferred_services": ["netflix", "prime_video", "hbo_max"],
            "disliked_genres": ["Romance", "Music"],
            "description": (
                "Night-owl binger who loves atmospheric slow-burn horror "
                "and cerebral thrillers. Appreciates strong cinematography "
                "over jump scares. Has a soft spot for A24 productions."
            ),
            "traits": ["night-owl-binger", "a24-devotee", "cinematography-appreciator"],
            "recommendation_focus": {
                "emphasize": ["atmospheric tension", "visual storytelling"],
                "avoid": ["torture porn", "found footage"],
            },
        },
        "interaction_history": {
            "recent_events": [
                {
                    "event_type": "swipe_right",
                    "content_ref": "tmdb:493922",
                    "imdb_id": "tt7784604",
                    "timestamp": "2026-02-16T23:15:00Z",
                },
                {
                    "event_type": "swipe_left",
                    "content_ref": "tmdb:346364",
                    "imdb_id": "tt3322364",
                    "timestamp": "2026-02-16T23:14:00Z",
                },
                {
                    "event_type": "watch_trailer",
                    "content_ref": "tmdb:493922",
                    "imdb_id": "tt7784604",
                    "timestamp": "2026-02-16T23:16:00Z",
                },
            ],
            "watched": ["tt7784604", "tt5052448"],
            "watchlist": ["tt8772262"],
            "exclude": [
                "tt7784604",
                "tt3322364",
                "tt5052448",
                "tt8772262",
                "tt4633694",
                "tt3235888",
            ],
        },
    }


@pytest.fixture
def query_only_request():
    """Search query with empty profile — user skipped onboarding."""
    return {
        "user_id": str(uuid4()),
        "query": "feel-good comedies for a rainy Sunday",
        "limit": 20,
        "region": "US",
        "profile": {
            "preferred_genres": [],
            "preferred_services": [],
            "disliked_genres": [],
            "description": None,
            "traits": [],
            "recommendation_focus": {},
        },
        "interaction_history": {
            "recent_events": [],
            "watched": [],
            "watchlist": [],
            "exclude": [],
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IMDB_ID_PATTERN = r"^tt\d+$"
