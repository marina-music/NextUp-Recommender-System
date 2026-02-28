# CineMatch External Recommender API — Overview

## What is CineMatch?

CineMatch is an AI-powered cross-platform movie discovery app. Users discover movies through a **swipe-based interface** (like Tinder for movies) — swipe right to add to watchlist, swipe left to skip. They can also search via natural language chat ("show me atmospheric horror from the 2020s") and browse their watchlist grouped by genre.

CineMatch runs on iOS, Android, web browsers, and TV platforms (Fire TV, Android TV).

## How the Recommender Fits In

```
User opens CineMatch
    |
    v
CineMatch collects:
  - Explicit preferences (genres, streaming services)
  - LLM-generated profile (narrative description + traits)
  - Interaction history (swipes, trailer views, watchlist actions)
    |
    v
CineMatch calls YOUR service:  POST /recommend
  -> Sends user profile + interaction history + exclusion set
    |
    v
YOUR service returns:
  - Ranked movie recommendations (IMDB IDs + metadata)
    |
    v
CineMatch resolves IMDB IDs to full movie details (posters, cast, trailers)
and displays them as swipeable cards to the user
```

Your service is one of multiple recommendation strategies. CineMatch's A-B testing system routes a percentage of traffic to your service and measures how users respond (swipe rates, watchlist adds, trailer views).

## Key Concept: The Swiped Exclusion Set

Every request includes an `exclude` array of IMDB IDs. These are movies the user has already interacted with (swiped right or left). **Your service must never return a movie whose IMDB ID is in the exclude list.** This is the most important rule — returning already-seen movies is a terrible user experience.

The exclude list grows over time. A new user might have 0 entries; a power user could have 500+.

## Authentication

All requests use **HTTP Basic Authentication**.

```
Authorization: Basic base64(username:password)
```

Credentials will be provided separately. Every request must include this header.

## Quick Start

1. Read `api-contract.md` for the full endpoint specification with sample requests/responses
2. Read `data-dictionary.md` for reference tables (genres, streaming services, event types)
3. Implement `POST /recommend` following the contract
4. Run the test suite to validate your implementation:

```bash
cd docs/api/test-suite
pip install pytest httpx
RECOMMENDER_URL=http://localhost:9000 \
RECOMMENDER_USER=test \
RECOMMENDER_PASS=test \
pytest -v
```

## Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This overview |
| `api-contract.md` | Full endpoint spec with sample JSON |
| `data-dictionary.md` | Reference tables for genres, services, events, profile fields |
| `test-suite/` | Pytest contract tests you can run against your implementation |
| `external-recommender-spec.yaml` | OpenAPI 3.0.3 machine-readable spec |
