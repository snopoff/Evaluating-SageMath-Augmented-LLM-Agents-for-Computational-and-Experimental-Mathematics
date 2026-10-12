import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBLEMS_PATH = ROOT / "data/processed/normalized_realmath.json"
ANSWER_TYPES = ("number", "expression")


DocumentShape = Literal["array", "jsonl", "object_stream", "results", "problems", "single"]


@dataclass(frozen=True)
class PredictionDocument:
    rows: list[dict[str, Any]]
    shape: DocumentShape
    payload: dict[str, Any] | None = None


def split_predictions_by_answer_type(
    input_path: str | Path,
    problems_path: str | Path = DEFAULT_PROBLEMS_PATH,
    *,
    log_progress: bool = False,
    strict: bool = False,
) -> dict[str, Path]:
    source = Path(input_path)
    document = _load_prediction_document(source)
    answer_types_by_id = _load_answer_types_by_id(Path(problems_path))
    if log_progress:
        print("loaded")

    filtered_rows: dict[str, list[dict[str, Any]]] = {answer_type: [] for answer_type in ANSWER_TYPES}
    missing_ids: set[str] = set()

    for index, row in enumerate(document.rows):
        problem_id = row.get("id")
        if not isinstance(problem_id, str) or not problem_id:
            missing_ids.add(f"row-{index + 1:05d}")
            continue

        answer_type = answer_types_by_id.get(problem_id)
        if answer_type in filtered_rows:
            filtered_rows[answer_type].append(_row_with_answer_type(row, answer_type))
        elif answer_type is None:
            missing_ids.add(problem_id)

    if strict and missing_ids:
        preview = ", ".join(sorted(missing_ids)[:10])
        suffix = "" if len(missing_ids) <= 10 else f", ... ({len(missing_ids)} total)"
        raise ValueError(f"No answer type found for prediction IDs: {preview}{suffix}")
    if log_progress:
        print("processed")

    output_paths = {
        "all": source.with_name(f"{source.stem}_all.json"),
        **{answer_type: source.with_name(f"{source.stem}_{answer_type}.json") for answer_type in ANSWER_TYPES},
    }
    _write_prediction_document(document, output_paths["all"], [_row_with_answer_type(row, answer_types_by_id.get(str(row.get("id", "")), "")) for row in document.rows])
    for answer_type, output_path in output_paths.items():
        if answer_type == "all":
            continue
        _write_prediction_document(document, output_path, filtered_rows[answer_type])
    if log_progress:
        print("saved")

    return output_paths


def _row_with_answer_type(row: dict[str, Any], answer_type: str) -> dict[str, Any]:
    if not answer_type:
        return dict(row)
    updated_row = dict(row)
    updated_row["answer_type"] = answer_type
    return updated_row


def _load_prediction_document(path: Path) -> PredictionDocument:
    if path.suffix == ".jsonl":
        return PredictionDocument(rows=_load_jsonl_rows(path), shape="jsonl")

    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = list(_iter_json_object_stream(text))
        return PredictionDocument(rows=rows, shape="object_stream")

    if isinstance(payload, list):
        return PredictionDocument(rows=_dict_rows(payload), shape="array")

    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return PredictionDocument(rows=_dict_rows(payload["results"]), shape="results", payload=payload)
        if isinstance(payload.get("problems"), list):
            return PredictionDocument(rows=_dict_rows(payload["problems"]), shape="problems", payload=payload)
        return PredictionDocument(rows=[payload], shape="single")

    raise ValueError(f"Unsupported JSON payload in {path}")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(payload)
    return rows


def _iter_json_object_stream(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break

        payload, index = decoder.raw_decode(text, index)
        if not isinstance(payload, dict):
            raise ValueError("Expected every JSON stream item to be an object")
        rows.append(payload)

    return rows


def _dict_rows(payload: list[Any]) -> list[dict[str, Any]]:
    return [row for row in payload if isinstance(row, dict)]


def _write_prediction_document(document: PredictionDocument, path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Any
    if document.shape in {"results", "problems"} and document.payload is not None:
        payload = dict(document.payload)
        payload[document.shape] = rows
    else:
        payload = rows

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _load_answer_types_by_id(problems_path: Path) -> dict[str, str]:
    answer_types_by_id: dict[str, str] = {}

    if problems_path.exists():
        for row in _load_reference_rows(problems_path):
            problem_id = row.get("id")
            answer_type = _answer_type_from_row(row)
            if isinstance(problem_id, str) and answer_type in ANSWER_TYPES:
                _set_answer_type(answer_types_by_id, problem_id, answer_type, problems_path)

    for answer_type in ANSWER_TYPES:
        split_path = problems_path.with_name(f"normalized_problems_by_type_{answer_type}.json")
        if not split_path.exists():
            continue
        for row in _load_reference_rows(split_path):
            problem_id = row.get("id")
            if isinstance(problem_id, str):
                _set_answer_type(answer_types_by_id, problem_id, answer_type, split_path)

    if not answer_types_by_id:
        raise ValueError(f"No number/expression answer types found from {problems_path}")

    return answer_types_by_id


def _load_reference_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    if isinstance(payload, list):
        return _dict_rows(payload)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return _dict_rows(payload["results"])
    if isinstance(payload, dict) and isinstance(payload.get("problems"), list):
        return _dict_rows(payload["problems"])
    if isinstance(payload, dict):
        return [payload]
    return []


def _answer_type_from_row(row: dict[str, Any]) -> str | None:
    answer_type = row.get("answer_type")
    if answer_type in ANSWER_TYPES:
        return str(answer_type)

    llm_payload = row.get("llm")
    if isinstance(llm_payload, dict):
        answer_type = llm_payload.get("answer_type")
        if answer_type in ANSWER_TYPES:
            return str(answer_type)

    return None


def _set_answer_type(mapping: dict[str, str], problem_id: str, answer_type: str, source_path: Path) -> None:
    existing = mapping.get(problem_id)
    if existing is not None and existing != answer_type:
        raise ValueError(f"Conflicting answer types for problem ID {problem_id}: {existing!r} and {answer_type!r} from {source_path}")
    mapping[problem_id] = answer_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Split prediction rows into number-only and expression-only files.")
    parser.add_argument("input_path", help="Path to the input predictions file.")
    parser.add_argument(
        "--problems-path",
        default=str(DEFAULT_PROBLEMS_PATH),
        help="Path to normalized problems metadata. Defaults to data/processed/normalized_realmath.json.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a prediction ID is missing from the answer-type metadata.",
    )
    args = parser.parse_args()

    output_paths = split_predictions_by_answer_type(args.input_path, args.problems_path, log_progress=True, strict=args.strict)
    print(json.dumps({answer_type: str(path) for answer_type, path in output_paths.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
