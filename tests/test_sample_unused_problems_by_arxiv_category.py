import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark.sample_unused_problems_by_arxiv_category import sample_unused_problems_by_arxiv_category


class SampleUnusedProblemsByArxivCategoryTest(unittest.TestCase):
    def test_sampler_filters_by_primary_category_and_excludes_used_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            normalized_path = tmp_path / "normalized.json"
            tags_path = tmp_path / "tags.json"
            used_path = tmp_path / "used.json"
            output_path = tmp_path / "output.json"
            stats_path = tmp_path / "stats.json"

            normalized_path.write_text(
                json.dumps(
                    [
                        {"id": "p1", "question": "q1"},
                        {"id": "p2", "question": "q2"},
                        {"id": "p3", "question": "q3"},
                        {"id": "p4", "question": "q4"},
                    ]
                ),
                encoding="utf-8",
            )
            tags_path.write_text(
                json.dumps(
                    [
                        {"id": "p1", "arxiv_id": "a1", "link": "l1", "categories": ["math.AG", "math.NT"]},
                        {"id": "p2", "arxiv_id": "a2", "link": "l2", "categories": ["math.GT"]},
                        {"id": "p3", "arxiv_id": "a3", "link": "l3", "categories": ["math.CO", "math.AG"]},
                        {"id": "p4", "arxiv_id": "a4", "link": "l4", "categories": ["math.AG"]},
                    ]
                ),
                encoding="utf-8",
            )
            used_path.write_text(json.dumps([{"id": "p2"}]), encoding="utf-8")

            sample_unused_problems_by_arxiv_category(
                normalized_path=normalized_path,
                tags_path=tags_path,
                used_path=used_path,
                output_path=output_path,
                stats_path=stats_path,
                target_categories=["math.AG", "math.GT"],
            )

            output_rows = json.loads(output_path.read_text(encoding="utf-8"))
            stats = json.loads(stats_path.read_text(encoding="utf-8"))

        self.assertEqual([row["id"] for row in output_rows], ["p1", "p4"])
        self.assertEqual([row["arxiv_top_category"] for row in output_rows], ["math.AG", "math.AG"])
        self.assertEqual(stats["eligible_unused_problem_count"], 2)
        self.assertEqual(stats["selected_problem_count"], 2)

    def test_sampler_can_limit_to_target_count_per_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            normalized_path = tmp_path / "normalized.json"
            tags_path = tmp_path / "tags.json"
            used_path = tmp_path / "used.json"
            output_path = tmp_path / "output.json"
            stats_path = tmp_path / "stats.json"

            normalized_path.write_text(
                json.dumps([{"id": "p1"}, {"id": "p2"}, {"id": "p3"}, {"id": "p4"}]),
                encoding="utf-8",
            )
            tags_path.write_text(
                json.dumps(
                    [
                        {"id": "p1", "categories": ["math.AG"]},
                        {"id": "p2", "categories": ["math.AG"]},
                        {"id": "p3", "categories": ["math.AG"]},
                        {"id": "p4", "categories": ["math.AG"]},
                    ]
                ),
                encoding="utf-8",
            )
            used_path.write_text(json.dumps([{"id": "p1"}]), encoding="utf-8")

            sample_unused_problems_by_arxiv_category(
                normalized_path=normalized_path,
                tags_path=tags_path,
                used_path=used_path,
                output_path=output_path,
                stats_path=stats_path,
                target_categories=["math.AG"],
                target_count_per_category=3,
            )

            output_rows = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual([row["id"] for row in output_rows], ["p2", "p3"])
