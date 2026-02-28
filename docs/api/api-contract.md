# CineMatch External Recommender API — Endpoint Specification

## `POST /recommend`

Returns personalized movie recommendations for a user based on their profile and interaction history.

### Headers

| Header | Value | Required |
|--------|-------|----------|
| `Content-Type` | `application/json` | Yes |
| `Authorization` | `Basic base64(username:password)` | Yes |

### Request Schema

```json
{
    "user_id": "string (UUID)",
    "query": "string or null",
    "limit": 20,
    "region": "US",
    "profile": { ... },
    "interaction_history": { ... }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | UUID string | Yes | Unique user identifier. Stable across sessions. |
| `query` | string or null | No | Natural language search text if user typed a query (e.g., "90s action movies"). Null for general recommendations. |
| `limit` | integer | Yes | Maximum number of recommendations to return. Default: 20, max: 50. |
| `region` | string | No | ISO 3166-1 alpha-2 country code. Affects streaming availability. Default: "US". |
| `profile` | object | No | User's taste profile. See Profile section below. May be empty for new users. |
| `interaction_history` | object | No | User's interaction history. See Interaction History section below. May be empty for new users. |

### Profile Object

Contains the user's explicit preferences and LLM-generated profile data. All fields are optional — new users may have none of these set.

```json
{
    "preferred_genres": ["Horror", "Sci-Fi", "Thriller"],
    "preferred_services": ["netflix", "prime_video"],
    "disliked_genres": ["Romance"],
    "description": "Night-owl binger who loves atmospheric slow-burn horror and cerebral sci-fi. Appreciates strong cinematography over jump scares. Watches mostly alone late at night. Has a soft spot for 80s practical effects but equally enjoys modern indie horror like A24 productions.",
    "traits": ["night-owl-binger", "indie-film-buff", "cinematography-appreciator"],
    "recommendation_focus": {
        "emphasize": ["atmospheric cinematography", "slow-burn tension", "practical effects"],
        "avoid": ["excessive gore", "romantic subplots", "found footage"]
    }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `preferred_genres` | string[] | Genre labels the user selected during onboarding. See data dictionary for full list. |
| `preferred_services` | string[] | Streaming service keys the user has access to. See data dictionary for full list. |
| `disliked_genres` | string[] | Genres the user explicitly dislikes. |
| `description` | string or null | LLM-generated narrative profile (~100 words) from onboarding interview. Null if user skipped interview. This is the richest signal about user taste. |
| `traits` | string[] | LLM-generated trait tags (3-5 tags). Examples: "night-owl-binger", "family-viewer", "indie-film-buff". |
| `recommendation_focus` | object | LLM-generated guidance: `emphasize` (what to lean into) and `avoid` (what to steer away from). Both are string arrays. |

### Interaction History Object

Contains the user's behavioral data — what they've done in the app.

```json
{
    "recent_events": [
        {
            "event_type": "swipe_right",
            "content_ref": "tmdb:508947",
            "imdb_id": "tt1375666",
            "timestamp": "2026-02-16T10:30:00Z"
        },
        {
            "event_type": "swipe_left",
            "content_ref": "tmdb:299534",
            "imdb_id": "tt4154796",
            "timestamp": "2026-02-16T10:29:00Z"
        },
        {
            "event_type": "watch_trailer",
            "content_ref": "tmdb:508947",
            "imdb_id": "tt1375666",
            "timestamp": "2026-02-16T10:31:00Z"
        },
        {
            "event_type": "card_flip",
            "content_ref": "tmdb:508947",
            "imdb_id": "tt1375666",
            "timestamp": "2026-02-16T10:30:30Z"
        }
    ],
    "watched": ["tt0137523", "tt0468569"],
    "watchlist": ["tt1375666", "tt0111161"],
    "exclude": ["tt1375666", "tt4154796", "tt0137523", "tt0468569", "tt0111161"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `recent_events` | object[] | Last 100 user interactions, newest first. Each has `event_type`, `content_ref` (tmdb:id format), `imdb_id`, and `timestamp`. See data dictionary for all event types. |
| `watched` | string[] | IMDB IDs of movies the user has marked as "already watched". |
| `watchlist` | string[] | IMDB IDs of movies currently in the user's watchlist (want to watch). |
| `exclude` | string[] | **CRITICAL:** IMDB IDs of ALL movies the user has already interacted with (swiped in either direction, watched, in watchlist). Your response MUST NOT contain any IMDB ID from this list. |

### Behavioral Signals

The interaction history encodes taste signals your recommender should leverage:

| Signal | What It Means |
|--------|--------------|
| `swipe_right` on a movie | User is interested — similar movies are good candidates |
| `swipe_left` on a movie | User is not interested — avoid similar movies |
| `card_flip` (viewed details) | Curiosity signal — user wanted to learn more |
| `watch_trailer` | Strong interest signal — user invested time watching the trailer |
| `mark_watched` | User has already seen this movie — don't recommend it, but use it as a taste indicator |
| `tap_streaming_link` | User may have actually watched this — strongest positive signal |

---

## Response Schema (Success — 200)

```json
{
    "recommendations": [
        {
            "imdb_id": "tt28607951",
            "title": "The Substance",
            "year": "2024",
            "why_recommended": "Matches your love of body horror with strong cinematography",
            "confidence": 0.87
        },
        {
            "imdb_id": "tt6263850",
            "title": "Nope",
            "year": "2022",
            "why_recommended": "Jordan Peele's atmospheric sci-fi horror aligns with your slow-burn preference",
            "confidence": 0.82
        }
    ],
    "response_message": "Based on your taste for atmospheric horror with strong visuals, here are some picks you might enjoy.",
    "usage": {
        "provider": "openai",
        "model": "ft:gpt-4o-mini:cinematch",
        "input_tokens": 1200,
        "output_tokens": 800,
        "cost_usd": 0.00068,
        "latency_ms": 340
    }
}
```

### Recommendation Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `imdb_id` | string | Yes | IMDB ID in format `tt` followed by digits (e.g., `tt1375666`). This is the canonical identifier. CineMatch resolves this to full movie details (poster, cast, trailer) on its side. |
| `title` | string | Yes | Movie title. Used for logging/debugging — CineMatch will override with TMDB data. |
| `year` | string | Yes | Release year as string (e.g., "2024"). |
| `why_recommended` | string | No | Human-readable explanation shown to the user on the card back. Keep under 100 characters. |
| `confidence` | number | No | 0.0 to 1.0 — how confident your model is in this recommendation. Used for ranking when blending with other strategies. |

### Response Message

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response_message` | string | Yes | A conversational summary shown at the top of the results (e.g., "Here are some atmospheric horror picks for you"). Keep under 200 characters. |

### Usage Report (Optional)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `usage` | object | No | If your service uses an LLM internally, report token usage here. CineMatch logs this for cost tracking. Omit entirely if you don't use an LLM. |
| `usage.provider` | string | No | e.g., "openai", "anthropic" |
| `usage.model` | string | No | e.g., "gpt-4o-mini", "ft:gpt-4o-mini:cinematch" |
| `usage.input_tokens` | integer | No | Input tokens consumed |
| `usage.output_tokens` | integer | No | Output tokens generated |
| `usage.cost_usd` | number | No | Estimated cost in USD |
| `usage.latency_ms` | integer | No | LLM call latency in milliseconds |

---

## Response Quality Rules

Your recommendations must satisfy these rules:

1. **No excluded movies** — Never return an `imdb_id` that appears in `interaction_history.exclude`. This is the most critical rule.
2. **No duplicates** — Each `imdb_id` in the response must be unique.
3. **Minimum 5 results** — Return at least 5 recommendations when possible. If you genuinely can't find 5 suitable movies, return what you have.
4. **Valid IMDB IDs** — All `imdb_id` values must match the pattern `tt\d+` (e.g., `tt1375666`). Invalid IDs cause resolution failures on CineMatch's side.
5. **Respect the limit** — Never return more than `limit` recommendations.
6. **Diversity** — Aim for variety. Don't return 20 movies from the same franchise or year. Mix genres, decades, and styles within the user's taste profile.
7. **Recency** — Favor movies the user is likely to find on streaming services. A recommendation for an obscure 1940s film that isn't available anywhere provides poor UX.

---

## Sample Requests

### Sample 1: New User (Minimal History)

A user who just completed onboarding — selected genres and services but hasn't swiped yet.

**Request:**
```json
{
    "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "query": null,
    "limit": 20,
    "region": "US",
    "profile": {
        "preferred_genres": ["Action", "Sci-Fi"],
        "preferred_services": ["netflix", "disney_plus"],
        "disliked_genres": [],
        "description": null,
        "traits": [],
        "recommendation_focus": {}
    },
    "interaction_history": {
        "recent_events": [],
        "watched": [],
        "watchlist": [],
        "exclude": []
    }
}
```

**Expected behavior:** Return broadly popular action and sci-fi movies available on Netflix and Disney+. Since there's no interaction history, rely on genre preferences.

### Sample 2: Power User (Rich History)

A user with a detailed profile and extensive swipe history.

**Request:**
```json
{
    "user_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "query": "something like Hereditary but less depressing",
    "limit": 15,
    "region": "CA",
    "profile": {
        "preferred_genres": ["Horror", "Thriller", "Mystery"],
        "preferred_services": ["netflix", "crave", "prime_video"],
        "disliked_genres": ["Romance", "Music"],
        "description": "Night-owl binger who loves atmospheric slow-burn horror and cerebral thrillers. Appreciates strong cinematography over jump scares. Has a soft spot for A24 productions and Korean cinema. Watches mostly alone late at night after the kids are asleep.",
        "traits": ["night-owl-binger", "a24-devotee", "cinematography-appreciator", "korean-cinema-fan"],
        "recommendation_focus": {
            "emphasize": ["atmospheric tension", "unreliable narrators", "visual storytelling"],
            "avoid": ["torture porn", "found footage", "teen slashers"]
        }
    },
    "interaction_history": {
        "recent_events": [
            {"event_type": "swipe_right", "content_ref": "tmdb:493922", "imdb_id": "tt7784604", "timestamp": "2026-02-16T23:15:00Z"},
            {"event_type": "watch_trailer", "content_ref": "tmdb:493922", "imdb_id": "tt7784604", "timestamp": "2026-02-16T23:16:00Z"},
            {"event_type": "swipe_left", "content_ref": "tmdb:346364", "imdb_id": "tt3322364", "timestamp": "2026-02-16T23:14:00Z"},
            {"event_type": "swipe_right", "content_ref": "tmdb:530385", "imdb_id": "tt8772262", "timestamp": "2026-02-16T23:12:00Z"},
            {"event_type": "card_flip", "content_ref": "tmdb:530385", "imdb_id": "tt8772262", "timestamp": "2026-02-16T23:12:30Z"}
        ],
        "watched": ["tt7784604", "tt5052448", "tt1457767"],
        "watchlist": ["tt8772262", "tt10366206"],
        "exclude": [
            "tt7784604", "tt3322364", "tt8772262", "tt5052448",
            "tt1457767", "tt10366206", "tt4633694", "tt3235888",
            "tt1220634", "tt2024544", "tt0264464", "tt0256524"
        ]
    }
}
```

**Expected behavior:** Atmospheric horror/thriller recommendations avoiding the 12 excluded movies. Lean into A24 style, Korean cinema, and visual storytelling. The query "something like Hereditary but less depressing" should guide toward eerie/creepy rather than grief-heavy horror. Region is Canada, so prioritize content on Netflix, Crave, and Prime Video (CA catalogs).

### Sample 3: Search Query, No Profile

A user who skipped onboarding but typed a search query.

**Request:**
```json
{
    "user_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
    "query": "feel-good comedies for a rainy Sunday",
    "limit": 20,
    "region": "US",
    "profile": {
        "preferred_genres": [],
        "preferred_services": [],
        "disliked_genres": [],
        "description": null,
        "traits": [],
        "recommendation_focus": {}
    },
    "interaction_history": {
        "recent_events": [],
        "watched": [],
        "watchlist": [],
        "exclude": []
    }
}
```

**Expected behavior:** Rely entirely on the query text. Return crowd-pleasing comedies with a warm/cozy tone. Without service preferences, don't filter by streaming availability.

---

## Error Responses

All error responses use this format:

```json
{
    "error": "Human-readable error message",
    "code": "ERROR_CODE"
}
```

### Error Codes

| Code | HTTP Status | When to Use |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed JSON, missing required fields, invalid UUID format |
| `UNAUTHORIZED` | 401 | Missing or invalid Basic Auth credentials |
| `INSUFFICIENT_DATA` | 422 | Not enough information to generate recommendations (e.g., no query AND no profile AND no history) |
| `TIMEOUT` | 503 | Your internal processing exceeded your own timeout |
| `SERVICE_UNAVAILABLE` | 503 | Your service is temporarily unable to fulfill requests |

### Error Response Examples

**400 — Invalid Request:**
```json
{
    "error": "Field 'user_id' is required and must be a valid UUID",
    "code": "INVALID_REQUEST"
}
```

**401 — Unauthorized:**
```json
{
    "error": "Invalid credentials",
    "code": "UNAUTHORIZED"
}
```

**422 — Insufficient Data:**
```json
{
    "error": "Cannot generate recommendations: no query, profile, or interaction history provided",
    "code": "INSUFFICIENT_DATA"
}
```

**503 — Service Unavailable:**
```json
{
    "error": "Recommendation model is currently loading, please retry in 30 seconds",
    "code": "SERVICE_UNAVAILABLE"
}
```

---

## Timeout

CineMatch will wait **10 seconds** for a response. If your service doesn't respond within 10 seconds, CineMatch falls back to its built-in recommendation strategy. Aim for response times under 3 seconds.

---

## `GET /usage-report` (Future)

Monthly usage statistics endpoint. Not required for initial implementation. Will be specified separately when needed.
