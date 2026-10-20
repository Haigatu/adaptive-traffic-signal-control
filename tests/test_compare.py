import json
from pathlib import Path
import tempfile
import unittest
from atsc.compare import collect_pair


class PairTests(unittest.TestCase):
    def test_rejects_different_demand(self):
        with tempfile.TemporaryDirectory(dir=".", ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            for name in ("no_forecast", "forecast"):
                child = root / name
                child.mkdir()
                (child / "test.rou.xml").write_text("<routes/>")
                (child / "summary.json").write_text(json.dumps({
                    "status": "completed", "seed": 1, "saturation": .7,
                    "warmup": 0, "duration": 30, "metrics": {"elapsed_s": 30},
                }))
            self.assertEqual(collect_pair(root)["status"], "completed")
            (root / "forecast/test.rou.xml").write_text("<different/>")
            with self.assertRaisesRegex(ValueError, "route_sha256"):
                collect_pair(root)
