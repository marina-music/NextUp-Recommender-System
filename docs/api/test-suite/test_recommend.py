"""Contract tests for POST /recommend.

These tests validate that the external recommender service conforms to
the CineMatch API contract. They test the HTTP interface, not internals.

Run:
    RECOMMENDER_URL=http://localhost:9000 \
    RECOMMENDER_USER=test \
    RECOMMENDER_PASS=test \
    pytest test_recommend.py -v
"""

import re

from conftest import IMDB_ID_PATTERN


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestHappyPath:
    def test_new_user_returns_recommendations(self, client, new_user_request):
        """New user with genre preferences gets recommendations."""
        resp = client.post("/recommend", json=new_user_request)
        assert resp.status_code == 200

        data = resp.json()
        assert "recommendations" in data
        assert "response_message" in data
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) >= 1

    def test_power_user_returns_recommendations(self, client, power_user_request):
        """User with rich history gets personalized recommendations."""
        resp = client.post("/recommend", json=power_user_request)
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["recommendations"]) >= 1
        assert len(data["recommendations"]) <= power_user_request["limit"]

    def test_query_only_returns_recommendations(self, client, query_only_request):
        """User with just a search query (no profile) gets results."""
        resp = client.post("/recommend", json=query_only_request)
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["recommendations"]) >= 1

    def test_response_message_is_string(self, client, new_user_request):
        """Response message is a non-empty string."""
        resp = client.post("/recommend", json=new_user_request)
        data = resp.json()
        assert isinstance(data["response_message"], str)
        assert len(data["response_message"]) > 0


# ---------------------------------------------------------------------------
# Recommendation schema validation
# ---------------------------------------------------------------------------
class TestRecommendationSchema:
    def test_imdb_ids_are_valid(self, client, new_user_request):
        """All imdb_id values match the tt<digits> pattern."""
        resp = client.post("/recommend", json=new_user_request)
        data = resp.json()
        for rec in data["recommendations"]:
            assert "imdb_id" in rec, "Recommendation missing imdb_id"
            assert re.match(IMDB_ID_PATTERN, rec["imdb_id"]), (
                f"Invalid IMDB ID format: {rec['imdb_id']}"
            )

    def test_required_fields_present(self, client, new_user_request):
        """Each recommendation has required fields: imdb_id, title, year."""
        resp = client.post("/recommend", json=new_user_request)
        data = resp.json()
        for rec in data["recommendations"]:
            assert "imdb_id" in rec
            assert "title" in rec
            assert "year" in rec
            assert isinstance(rec["title"], str)
            assert isinstance(rec["year"], str)

    def test_confidence_in_valid_range(self, client, new_user_request):
        """If confidence is provided, it must be between 0.0 and 1.0."""
        resp = client.post("/recommend", json=new_user_request)
        data = resp.json()
        for rec in data["recommendations"]:
            if "confidence" in rec and rec["confidence"] is not None:
                assert 0.0 <= rec["confidence"] <= 1.0, (
                    f"Confidence out of range: {rec['confidence']}"
                )

    def test_no_duplicate_imdb_ids(self, client, power_user_request):
        """No duplicate IMDB IDs in the response."""
        resp = client.post("/recommend", json=power_user_request)
        data = resp.json()
        ids = [r["imdb_id"] for r in data["recommendations"]]
        assert len(ids) == len(set(ids)), (
            f"Duplicate IMDB IDs found: {[x for x in ids if ids.count(x) > 1]}"
        )


# ---------------------------------------------------------------------------
# Exclusion set — CRITICAL
# ---------------------------------------------------------------------------
class TestExclusionSet:
    def test_excluded_movies_not_returned(self, client, power_user_request):
        """Recommendations must not contain any IMDB ID from the exclude list."""
        exclude_set = set(
            power_user_request["interaction_history"]["exclude"]
        )
        resp = client.post("/recommend", json=power_user_request)
        data = resp.json()

        returned_ids = {r["imdb_id"] for r in data["recommendations"]}
        overlap = returned_ids & exclude_set
        assert len(overlap) == 0, (
            f"Excluded movies were returned: {overlap}"
        )

    def test_large_exclusion_set(self, client, new_user_request):
        """Service handles a large exclusion set without errors."""
        # Simulate a power user who has swiped 200+ movies
        new_user_request["interaction_history"]["exclude"] = [
            f"tt{i:07d}" for i in range(100, 300)
        ]
        resp = client.post("/recommend", json=new_user_request)
        assert resp.status_code == 200

        data = resp.json()
        exclude_set = set(new_user_request["interaction_history"]["exclude"])
        returned_ids = {r["imdb_id"] for r in data["recommendations"]}
        assert len(returned_ids & exclude_set) == 0


# ---------------------------------------------------------------------------
# Limit enforcement
# ---------------------------------------------------------------------------
class TestLimits:
    def test_respects_limit(self, client, new_user_request):
        """Does not return more recommendations than requested."""
        new_user_request["limit"] = 5
        resp = client.post("/recommend", json=new_user_request)
        data = resp.json()
        assert len(data["recommendations"]) <= 5

    def test_small_limit(self, client, new_user_request):
        """Handles a limit of 1."""
        new_user_request["limit"] = 1
        resp = client.post("/recommend", json=new_user_request)
        data = resp.json()
        assert len(data["recommendations"]) <= 1


# ---------------------------------------------------------------------------
# Optional usage report
# ---------------------------------------------------------------------------
class TestUsageReport:
    def test_usage_field_is_valid_if_present(self, client, new_user_request):
        """If usage is included, it has the expected structure."""
        resp = client.post("/recommend", json=new_user_request)
        data = resp.json()

        if "usage" in data and data["usage"] is not None:
            usage = data["usage"]
            # All fields are optional, but if present they should be correct types
            if "provider" in usage:
                assert isinstance(usage["provider"], str)
            if "model" in usage:
                assert isinstance(usage["model"], str)
            if "input_tokens" in usage:
                assert isinstance(usage["input_tokens"], int)
            if "output_tokens" in usage:
                assert isinstance(usage["output_tokens"], int)
            if "cost_usd" in usage:
                assert isinstance(usage["cost_usd"], (int, float))
            if "latency_ms" in usage:
                assert isinstance(usage["latency_ms"], int)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
class TestAuthentication:
    def test_missing_auth_returns_401(self, unauthed_client, new_user_request):
        """Request without credentials returns 401."""
        resp = unauthed_client.post("/recommend", json=new_user_request)
        assert resp.status_code == 401

    def test_wrong_credentials_returns_401(self, new_user_request):
        """Request with wrong credentials returns 401."""
        import httpx
        from conftest import BASE_URL

        with httpx.Client(
            base_url=BASE_URL,
            auth=("wrong_user", "wrong_pass"),
            timeout=15.0,
        ) as bad_client:
            resp = bad_client.post("/recommend", json=new_user_request)
            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Invalid requests
# ---------------------------------------------------------------------------
class TestInvalidRequests:
    def test_missing_user_id_returns_400(self, client):
        """Missing required user_id field returns 400."""
        resp = client.post("/recommend", json={"limit": 20})
        assert resp.status_code in (400, 422)

    def test_invalid_json_returns_400(self, client):
        """Malformed JSON body returns 400."""
        resp = client.post(
            "/recommend",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)

    def test_invalid_uuid_returns_400(self, client):
        """Invalid UUID format returns 400."""
        resp = client.post(
            "/recommend",
            json={"user_id": "not-a-uuid", "limit": 20},
        )
        assert resp.status_code in (400, 422)
