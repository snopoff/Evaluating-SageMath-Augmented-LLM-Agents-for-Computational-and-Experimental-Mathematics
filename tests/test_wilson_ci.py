import json
import tempfile
import unittest
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.benchmark.wilson_ci import summarize_result_rows, wilson_interval, wilson_summary  # noqa: E402


class WilsonCITests(unittest.TestCase):
    def test_wilson_interval_matches_known_24_of_60_value(self) -> None:
        lower_bound, upper_bound = wilson_interval(24, 60)
        self.assertAlmostEqual(lower_bound, 0.286, places=3)
        self.assertAlmostEqual(upper_bound, 0.526, places=3)

    def test_summary_from_result_files_aggregates_json_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_path = tmp_path / "rows.json"
            jsonl_path = tmp_path / "rows.jsonl"
            json_path.write_text(
                json.dumps(
                    [
                        {"id": "a", "correct": True},
                        {"id": "b", "correct": False},
                    ]
                ),
                encoding="utf-8",
            )
            jsonl_path.write_text(
                json.dumps({"id": "c", "correct": True})
                + "\n"
                + json.dumps({"id": "d", "correct": True})
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_result_rows(input_paths=(json_path, jsonl_path))

            self.assertEqual(summary.correct, 3)
            self.assertEqual(summary.rows, 4)
            self.assertAlmostEqual(summary.accuracy, 0.75)

    def test_summary_can_filter_rows_by_id_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result_path = tmp_path / "results.json"
            filter_path = tmp_path / "filter.jsonl"
            result_path.write_text(
                json.dumps(
                    [
                        {"id": "keep-1", "correct": True},
                        {"id": "keep-2", "correct": False},
                        {"id": "drop-1", "correct": True},
                    ]
                ),
                encoding="utf-8",
            )
            filter_path.write_text(
                json.dumps({"id": "keep-1"}) + "\n" + json.dumps({"id": "keep-2"}) + "\n",
                encoding="utf-8",
            )

            summary = summarize_result_rows(input_paths=(result_path,), id_filter_paths=(filter_path,))

            self.assertEqual(summary.correct, 1)
            self.assertEqual(summary.rows, 2)
            self.assertAlmostEqual(summary.accuracy, 0.5)

    def test_wilson_summary_rejects_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            wilson_summary(correct=5, rows=4)
