import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.benchmark.split_predictions_by_answer_type import split_predictions_by_answer_type


class SplitPredictionsByAnswerTypeTests(unittest.TestCase):
    def test_splits_json_array_using_inline_answer_type_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            predictions_path = tmp_path / "predictions.json"
            problems_path = tmp_path / "normalized.json"

            predictions_path.write_text(
                json.dumps(
                    [
                        {"id": "number-1", "prediction": "4"},
                        {"id": "expression-1", "prediction": "n + m"},
                        {"id": "formula-1", "prediction": "x = y"},
                    ]
                ),
                encoding="utf-8",
            )
            problems_path.write_text(
                json.dumps(
                    [
                        {"id": "number-1", "answer_type": "number"},
                        {"id": "expression-1", "answer_type": "expression"},
                        {"id": "formula-1", "answer_type": "formula"},
                    ]
                ),
                encoding="utf-8",
            )

            output_paths = split_predictions_by_answer_type(predictions_path, problems_path)

            self.assertEqual(output_paths["all"], tmp_path / "predictions_all.json")
            self.assertEqual(output_paths["number"], tmp_path / "predictions_number.json")
            self.assertEqual(output_paths["expression"], tmp_path / "predictions_expression.json")
            self.assertEqual(
                json.loads(output_paths["all"].read_text(encoding="utf-8")),
                [
                    {"id": "number-1", "prediction": "4", "answer_type": "number"},
                    {"id": "expression-1", "prediction": "n + m", "answer_type": "expression"},
                    {"id": "formula-1", "prediction": "x = y"},
                ],
            )
            self.assertEqual(
                json.loads(output_paths["number"].read_text(encoding="utf-8")),
                [{"id": "number-1", "prediction": "4", "answer_type": "number"}],
            )
            self.assertEqual(
                json.loads(output_paths["expression"].read_text(encoding="utf-8")),
                [{"id": "expression-1", "prediction": "n + m", "answer_type": "expression"}],
            )

    def test_falls_back_to_normalized_type_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            predictions_path = tmp_path / "run.json"
            problems_path = tmp_path / "normalized_problems.json"

            predictions_path.write_text(
                json.dumps([{"id": "n1"}, {"id": "e1"}]),
                encoding="utf-8",
            )
            problems_path.write_text(json.dumps([{"id": "n1"}, {"id": "e1"}]), encoding="utf-8")
            (tmp_path / "normalized_problems_by_type_number.json").write_text(json.dumps([{"id": "n1"}]), encoding="utf-8")
            (tmp_path / "normalized_problems_by_type_expression.json").write_text(json.dumps([{"id": "e1"}]), encoding="utf-8")

            output_paths = split_predictions_by_answer_type(predictions_path, problems_path)

            self.assertEqual(json.loads(output_paths["number"].read_text(encoding="utf-8")), [{"id": "n1", "answer_type": "number"}])
            self.assertEqual(json.loads(output_paths["expression"].read_text(encoding="utf-8")), [{"id": "e1", "answer_type": "expression"}])

    def test_converts_jsonl_input_to_json_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            predictions_path = tmp_path / "predictions.jsonl"
            problems_path = tmp_path / "normalized.json"

            predictions_path.write_text('{"id":"n1","value":1}\n{"id":"e1","value":2}\n', encoding="utf-8")
            problems_path.write_text(
                json.dumps([{"id": "n1", "answer_type": "number"}, {"id": "e1", "answer_type": "expression"}]),
                encoding="utf-8",
            )

            output_paths = split_predictions_by_answer_type(predictions_path, problems_path)

            self.assertEqual(output_paths["all"], tmp_path / "predictions_all.json")
            self.assertEqual(output_paths["number"], tmp_path / "predictions_number.json")
            self.assertEqual(output_paths["expression"], tmp_path / "predictions_expression.json")
            self.assertEqual(
                json.loads(output_paths["all"].read_text(encoding="utf-8")),
                [{"id": "n1", "value": 1, "answer_type": "number"}, {"id": "e1", "value": 2, "answer_type": "expression"}],
            )
            self.assertEqual(json.loads(output_paths["number"].read_text(encoding="utf-8")), [{"id": "n1", "value": 1, "answer_type": "number"}])
            self.assertEqual(json.loads(output_paths["expression"].read_text(encoding="utf-8")), [{"id": "e1", "value": 2, "answer_type": "expression"}])

    def test_cli_prints_required_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            predictions_path = tmp_path / "predictions.json"
            problems_path = tmp_path / "normalized.json"

            predictions_path.write_text(json.dumps([{"id": "n1"}]), encoding="utf-8")
            problems_path.write_text(json.dumps([{"id": "n1", "answer_type": "number"}]), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/split_predictions_by_answer_type.py",
                    str(predictions_path),
                    "--problems-path",
                    str(problems_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )

            output_lines = result.stdout.splitlines()
            self.assertEqual(output_lines[:3], ["loaded", "processed", "saved"])
            saved_paths = json.loads("\n".join(output_lines[3:]))
            self.assertEqual(saved_paths["all"], str(tmp_path / "predictions_all.json"))
            self.assertEqual(saved_paths["number"], str(tmp_path / "predictions_number.json"))
            self.assertEqual(saved_paths["expression"], str(tmp_path / "predictions_expression.json"))


if __name__ == "__main__":
    unittest.main()
