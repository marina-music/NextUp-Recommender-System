# CineMatch External Recommender API — Data Dictionary

## Genre Reference

These are the genres users can select during onboarding. Values in `profile.preferred_genres` and `profile.disliked_genres` use the **Label** column.

| Key | Label | TMDB ID |
|-----|-------|---------|
| `action` | Action | 28 |
| `adventure` | Adventure | 12 |
| `animation` | Animation | 16 |
| `comedy` | Comedy | 35 |
| `crime` | Crime | 80 |
| `documentary` | Documentary | 99 |
| `drama` | Drama | 18 |
| `family` | Family | 10751 |
| `fantasy` | Fantasy | 14 |
| `history` | History | 36 |
| `horror` | Horror | 27 |
| `music` | Music | 10402 |
| `mystery` | Mystery | 9648 |
| `romance` | Romance | 10749 |
| `sci_fi` | Sci-Fi | 878 |
| `thriller` | Thriller | 53 |
| `war` | War | 10752 |
| `western` | Western | 37 |

**Notes:**
- Genre labels are TMDB-aligned. The TMDB ID is provided in case your model uses TMDB's genre taxonomy.
- Users select a minimum of 3 genres during onboarding (or skip entirely).
- `disliked_genres` are genres the user has explicitly opted out of.

---

## Streaming Service Reference

These are the streaming services users can select. Values in `profile.preferred_services` use the **Key** column.

| Key | Label | TMDB Provider Names |
|-----|-------|-------------------|
| `netflix` | Netflix | Netflix, Netflix basic with Ads |
| `prime_video` | Prime Video | Amazon Prime Video, Amazon Prime Video with Ads, Amazon Video |
| `disney_plus` | Disney+ | Disney Plus |
| `youtube` | YouTube | YouTube |
| `hbo_max` | Max | Max, Max Amazon Channel, HBO Max |
| `hulu` | Hulu | Hulu |
| `apple_tv_plus` | Apple TV+ | Apple TV Plus, Apple TV+, Apple TV, Apple TV Store, Apple iTunes |
| `paramount_plus` | Paramount+ | Paramount Plus, Paramount+ Amazon Channel, Paramount+ |
| `peacock` | Peacock | Peacock, Peacock Premium |
| `plex` | Plex | Plex |
| `crave` | Crave | Crave, Crave Amazon Channel |
| `tubi` | Tubi | Tubi TV |
| `crunchyroll` | Crunchyroll | Crunchyroll |
| `mubi` | MUBI | MUBI |

**Notes:**
- The "TMDB Provider Names" column shows how these services appear in TMDB's watch provider API. One service can have multiple TMDB names (e.g., "Max" and "HBO Max" are the same service).
- When users select `prime_video`, they have access to Amazon Prime Video content — consider this when ranking recommendations.
- Streaming catalogs vary by `region`. A movie on Netflix US may not be on Netflix CA.

---

## Event Types

These are the interaction events sent in `interaction_history.recent_events`. Each event represents one user action in the app.

| Event Type | Trigger | Taste Signal | Key Metadata |
|------------|---------|-------------|--------------|
| `swipe_right` | User swipes right on a movie card | **Positive** — user wants this movie | `card_position`, `screen` |
| `swipe_left` | User swipes left on a movie card | **Negative** — user rejected this movie | `card_position`, `screen` |
| `card_flip` | User taps card to see details (cast, synopsis, trailer) | **Curiosity** — user wanted more info before deciding | `dwell_time_ms` |
| `watch_trailer` | User plays the movie trailer | **Strong positive** — invested time watching | `trailer_key`, `watch_duration_ms` |
| `mark_watched` | User marks a movie as "already watched" | **Neutral/positive** — confirms they've seen it; use as taste indicator | `source` |
| `mark_watch_soon` | User marks a movie as "watch soon" | **Strong positive** — explicit intent to watch | `source` |
| `add_to_watchlist` | User manually adds to watchlist | **Positive** — deliberate save for later | `added_from` |
| `remove_from_watchlist` | User removes from watchlist | **Weak negative** — lost interest or already watched | |
| `tap_streaming_link` | User clicks link to streaming service | **Strongest positive** — may actually be going to watch it now | `service_name`, `url` |
| `onboarding_complete` | User finishes onboarding flow | **Context** — not a taste signal, but indicates profile completeness | `steps_completed`, `skipped_interview` |
| `search_query` | User types a natural language search | **Context** — reveals current mood/intent | `query_text` |

### How to Use Events for Recommendations

**Signal strength (strongest to weakest):**
1. `tap_streaming_link` — user is about to watch
2. `watch_trailer` — user invested time watching preview
3. `mark_watch_soon` — explicit intent
4. `swipe_right` — user liked it
5. `card_flip` — user was curious
6. `swipe_left` — user rejected it (use to deprioritize similar movies)

**Temporal weighting:** Recent events are more indicative of current taste than older events. A `swipe_right` from today is more relevant than one from 2 weeks ago.

**Pattern recognition:** A user who `card_flip`s a movie but then `swipe_left`s it had initial interest but something on the detail screen turned them off. This is a nuanced signal.

---

## Event Object Structure

Each event in `recent_events` has this shape:

```json
{
    "event_type": "swipe_right",
    "content_ref": "tmdb:508947",
    "imdb_id": "tt1375666",
    "timestamp": "2026-02-16T10:30:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | One of the event types listed above |
| `content_ref` | string or null | Content reference in format `tmdb:<id>` (e.g., `tmdb:508947`). Null for non-content events like `onboarding_complete`. |
| `imdb_id` | string or null | IMDB ID (e.g., `tt1375666`). Null for non-content events. |
| `timestamp` | string (ISO 8601) | When the event occurred, in UTC |

**Note:** Events are ordered newest-first. The `recent_events` array contains the last 100 interactions.

---

## User Profile Fields

These fields come from the user's onboarding process and profile settings.

### `description` (string or null)

An LLM-generated narrative profile, approximately 100 words. Created through a multi-turn interview during onboarding. This is the richest single piece of information about the user's taste.

**Example:**
> "Night-owl binger who loves atmospheric slow-burn horror and cerebral sci-fi. Appreciates strong cinematography over jump scares. Watches mostly alone late at night after the kids are asleep. Has a soft spot for 80s practical effects but equally enjoys modern indie horror like A24 productions. Values original storytelling over franchise sequels. Occasionally ventures into dark comedy and psychological thrillers. Avoids anything with excessive gore or romantic subplots."

**When null:** User skipped the onboarding interview. Fall back to `preferred_genres` and interaction history.

### `traits` (string[])

LLM-generated tags summarizing the user's viewing personality. Typically 3-5 tags.

**Examples:**
- `"night-owl-binger"` — watches late at night, multiple movies per session
- `"family-viewer"` — watches with family, prefers age-appropriate content
- `"indie-film-buff"` — prefers independent/arthouse cinema
- `"cinematography-appreciator"` — values visual quality
- `"comfort-rewatcher"` — frequently rewatches favorites
- `"new-release-chaser"` — prefers recent releases
- `"korean-cinema-fan"` — enjoys Korean films

### `recommendation_focus` (object)

LLM-generated guidance on what to emphasize and avoid.

```json
{
    "emphasize": ["atmospheric tension", "unreliable narrators", "visual storytelling"],
    "avoid": ["torture porn", "found footage", "teen slashers"]
}
```

Both `emphasize` and `avoid` are string arrays with free-form descriptors. They may be empty or the entire object may be `{}`.

---

## Recommendation Output Fields

### `imdb_id` (string, required)

The canonical movie identifier. Format: `tt` followed by digits (e.g., `tt1375666` for Inception).

**Important:** CineMatch uses this ID to look up full movie details (poster, cast, synopsis, trailer, streaming availability) from its own cache/TMDB. If an IMDB ID doesn't resolve to a real movie, the recommendation is silently dropped. Use valid IMDB IDs only.

### `title` (string, required)

The movie title in English. Used for logging and debugging. CineMatch will override this with data from TMDB, so exact formatting doesn't matter — but it should be recognizable.

### `year` (string, required)

Release year as a 4-digit string (e.g., `"2024"`). Used for disambiguation when multiple movies share a title.

### `why_recommended` (string, optional)

A short, human-readable explanation displayed to the user on the movie card. Keep it concise — under 100 characters.

**Good:** "Matches your love of slow-burn psychological horror"
**Too long:** "Based on your preference for atmospheric horror films with strong cinematography and your recent positive interactions with A24 productions, this film's deliberate pacing and visual style should appeal to you"

### `confidence` (number, optional)

A value between 0.0 and 1.0 indicating how confident your model is in this recommendation. Used by CineMatch when blending results from multiple recommendation strategies.

- `0.9+` — Very high confidence, strong match
- `0.7-0.9` — Good match
- `0.5-0.7` — Decent match, worth showing
- `<0.5` — Weak match, filler

If you don't have a meaningful confidence score, omit this field rather than hardcoding 1.0.
