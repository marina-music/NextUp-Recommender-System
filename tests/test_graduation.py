"""Tests for graduation manager."""
import tempfile
from pathlib import Path


class TestGraduationManager:
    def test_record_interaction(self):
        from graduation import GraduationManager
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GraduationManager(
                queue_path=Path(tmpdir) / "queue.json",
                graduation_threshold=3,
                mamba_catalog=set(),
            )
            gm.record_interaction("tt001")
            gm.record_interaction("tt001")
            assert gm.get_interaction_count("tt001") == 2

    def test_graduation_on_threshold(self):
        from graduation import GraduationManager
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GraduationManager(
                queue_path=Path(tmpdir) / "queue.json",
                graduation_threshold=3,
                mamba_catalog=set(),
            )
            gm.record_interaction("tt001")
            gm.record_interaction("tt001")
            graduated = gm.record_interaction("tt001")
            assert graduated is True
            assert "tt001" in gm.get_pending_graduations()

    def test_no_graduation_for_mamba_movies(self):
        from graduation import GraduationManager
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GraduationManager(
                queue_path=Path(tmpdir) / "queue.json",
                graduation_threshold=3,
                mamba_catalog={"tt001"},
            )
            for _ in range(5):
                gm.record_interaction("tt001")
            assert "tt001" not in gm.get_pending_graduations()

    def test_persistence(self):
        from graduation import GraduationManager
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            gm1 = GraduationManager(
                queue_path=queue_path,
                graduation_threshold=2,
                mamba_catalog=set(),
            )
            gm1.record_interaction("tt001")
            gm1.record_interaction("tt001")
            gm1.save()

            gm2 = GraduationManager(
                queue_path=queue_path,
                graduation_threshold=2,
                mamba_catalog=set(),
            )
            assert "tt001" in gm2.get_pending_graduations()

    def test_threshold_trigger(self):
        from graduation import GraduationManager
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GraduationManager(
                queue_path=Path(tmpdir) / "queue.json",
                graduation_threshold=1,
                mamba_catalog=set(),
                retrain_on_graduation_count=3,
            )
            gm.record_interaction("tt001")
            gm.record_interaction("tt002")
            gm.record_interaction("tt003")
            assert gm.should_retrain_by_threshold() is True

    def test_mark_retrained(self):
        from graduation import GraduationManager
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GraduationManager(
                queue_path=Path(tmpdir) / "queue.json",
                graduation_threshold=1,
                mamba_catalog=set(),
            )
            gm.record_interaction("tt001")
            gm.mark_retrained(["tt001"], batch_id="batch_001")
            assert "tt001" not in gm.get_pending_graduations()
            assert len(gm.get_completed()) == 1
