import json
from pathlib import Path
import tempfile
import unittest

from atsc.demo_report import aggregate, build_report
from atsc.data import ROOT


class ReportTests(unittest.TestCase):
    def test_saved_comparison_uses_paired_scenarios_and_worst_red(self):
        rows = json.loads((ROOT / "artifacts/continuation/test_results.json").read_text())
        groups = aggregate(rows)
        self.assertEqual(len(groups), 15)
        for group in groups:
            sample = [r for r in rows if r["controller"] == group["controller"]
                      and r["saturation"] == group["saturation"]]
            self.assertEqual(group["n"], 4)
            self.assertEqual(group["max_without_green_s"],
                             max(r["max_without_green_s"] for r in sample))
            self.assertAlmostEqual(group["throughput"],
                                   sum(r["throughput"] for r in sample) / 4)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])
        with self.assertRaises(ValueError):
            aggregate(rows + [rows[0]])

    def test_report_embeds_current_run_separately_and_escapes_script(self):
        with tempfile.TemporaryDirectory(dir=ROOT, ignore_cleanup_errors=True) as folder:
            folder = Path(folder)
            (folder / "summary.json").write_text(json.dumps({
                "status": "interrupted", "duration": 150, "warmup": 0,
                "agent": "guarded", "seed": 1000, "saturation": 0.7,
                "error": "</script><script>alert(1)</script>",
            }), encoding="utf-8")
            (folder / "decisions.jsonl").write_text("", encoding="utf-8")
            report = build_report(folder)
            html = report.read_text(encoding="utf-8")
            self.assertIn("ATSC / Results", html)
            self.assertIn("Saved benchmark", html)
            self.assertNotIn("</script><script>alert", html)
            self.assertNotIn("https://", html)
            self.assertTrue(report.is_file())
