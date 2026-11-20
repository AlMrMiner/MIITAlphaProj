import tempfile
import unittest
from pathlib import Path

from rail_validator.process import render_queue


class QueueRenderTests(unittest.TestCase):
    def test_missing_candidate_displays_dash(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "queue.html"
            render_queue(target, [{
                "app_id": "Z0029604", "field": "weight_t",
                "original": "1079.7", "candidate": None,
                "reason": "Масса превышает грузоподъёмность",
                "confidence": None,
            }], {"pending": 1})
            html = target.read_text(encoding="utf-8")
            self.assertIn("—", html)
            self.assertNotIn("99.5%", html)
            self.assertNotIn("0.0%", html)

    def test_existing_candidate_retains_percentage(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "queue.html"
            render_queue(target, [{
                "app_id": "Z1", "field": "cargo_name",
                "original": "test", "candidate": "correct",
                "reason": "Название отличается",
                "confidence": .995,
            }], {})
            self.assertIn("99.5%", target.read_text(encoding="utf-8"))
