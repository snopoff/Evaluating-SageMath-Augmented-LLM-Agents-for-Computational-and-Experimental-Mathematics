from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class WilsonSummary:
    correct: int
    rows: int
    accuracy: float
    confidence_level: float
    lower_bound: float
    upper_bound: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "rows": self.rows,
            "accuracy": self.accuracy,
            "confidence_level": self.confidence_level,
            "wilson_lower_bound": self.lower_bound,
            "wilson_upper_bound": self.upper_bound,
        }


def wilson_interval(correct: int, rows: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if rows <= 0:
        raise ValueError("rows must be > 0")
    if correct < 0 or correct > rows:
        raise ValueError("correct must satisfy 0 <= correct <= rows")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must satisfy 0 < confidence_level < 1")

    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = correct / rows
    denominator = 1.0 + (z * z) / rows
    center = (proportion + (z * z) / (2.0 * rows)) / denominator
    margin = (
        z
        / denominator
        * ((proportion * (1.0 - proportion) / rows) + ((z * z) / (4.0 * rows * rows))) ** 0.5
    )
    return center - margin, center + margin


def wilson_summary(correct: int, rows: int, confidence_level: float = 0.95) -> WilsonSummary:
    lower_bound, upper_bound = wilson_interval(correct, rows, confidence_level=confidence_level)
    return WilsonSummary(
        correct=correct,
        rows=rows,
        accuracy=correct / rows,
        confidence_level=confidence_level,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def summarize_result_rows(
    *,
    input_paths: Sequence[Path],
    correct_fields: Sequence[str] = ("correct", "is_correct", "matches_reference"),
    limit: int = -1,
    id_filter_paths: Sequence[Path] = (),
    confidence_level: float = 0.95,
) -> WilsonSummary:
    if not input_paths:
        raise ValueError("input_paths must not be empty")

    id_filter = _load_id_filter(id_filter_paths)
    rows = list(iter_result_rows(input_paths=input_paths, limit=limit, id_filter=id_filter))
    if not rows:
        raise ValueError("No rows selected from the provided inputs.")

    correct = sum(1 for row in rows if _row_is_correct(row, correct_fields))
    return wilson_summary(correct=correct, rows=len(rows), confidence_level=confidence_level)


def iter_result_rows(
    *,
    input_paths: Sequence[Path],
    limit: int = -1,
    id_filter: set[str] | None = None,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    for input_path in input_paths:
        for row in _load_rows(Path(input_path), limit=-1):
            if id_filter is not None:
                row_id = row.get("id")
                if not isinstance(row_id, str) or row_id not in id_filter:
                    continue
            yield row
            emitted += 1
            if limit != -1 and emitted >= limit:
                return


def _row_is_correct(row: dict[str, Any], correct_fields: Sequence[str]) -> bool:
    for field_name in correct_fields:
        value = row.get(field_name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
    return False


def _load_id_filter(paths: Sequence[Path]) -> set[str] | None:
    if not paths:
        return None

    selected_ids: set[str] = set()
    for path in paths:
        for row in _load_rows(Path(path), limit=-1):
            row_id = row.get("id")
            if isinstance(row_id, str):
                selected_ids.add(row_id)
    return selected_ids


def _load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        return _load_json_rows(path, limit)
    return _load_jsonl_rows(path, limit)


def _load_jsonl_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
                if limit != -1 and len(rows) >= limit:
                    break
    return rows


def _load_json_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _load_object_stream_rows(text, limit)

    if isinstance(payload, dict) and isinstance(payload.get("problems"), list):
        rows = [row for row in payload["problems"] if isinstance(row, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        rows = [row for row in payload["results"] if isinstance(row, dict)]
    elif isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        rows = []

    if limit != -1:
        rows = rows[:limit]
    return rows


def _load_object_stream_rows(text: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    starts = _record_start_offsets(text)
    if not starts:
        return rows

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
            if limit != -1 and len(rows) >= limit:
                break

    return rows


def _record_start_offsets(text: str) -> list[int]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if re.match(r'^\{"id"\s*:', stripped):
            offsets.append(offset)
        elif stripped.strip() == "{":
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index < len(lines) and lines[next_index].lstrip().startswith('"id"'):
                offsets.append(offset)
        offset += len(line)
    return offsets
