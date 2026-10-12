from __future__ import annotations

import argparse
import json
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.benchmark.wilson_ci import summarize_result_rows, wilson_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a Wilson confidence interval either from raw counts "
            "or from benchmark result files (.json / .jsonl)."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input-path",
        dest="input_paths",
        action="append",
        help="Result file to read. May be passed multiple times; rows are aggregated across all files.",
    )
    mode.add_argument("--correct", type=int, help="Number of correct predictions.")

    parser.add_argument("--rows", type=int, help="Total number of rows. Required with --correct.")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level in (0, 1). Defaults to 0.95.",
    )
    parser.add_argument(
        "--correct-field",
        dest="correct_fields",
        action="append",
        help=(
            "Boolean correctness field to check in input rows. "
            "May be passed multiple times. Defaults to correct, is_correct, matches_reference."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of selected rows to count across all input files. Use -1 for all rows.",
    )
    parser.add_argument(
        "--id-filter-path",
        dest="id_filter_paths",
        action="append",
        help=(
            "Optional file whose row IDs define the subset to score. "
            "May be passed multiple times; IDs are unioned across all such files."
        ),
    )
    args = parser.parse_args()

    if args.correct is not None:
        if args.rows is None:
            parser.error("--rows is required when --correct is provided.")
        summary = wilson_summary(args.correct, args.rows, confidence_level=args.confidence_level)
        payload = summary.as_dict()
        payload["mode"] = "counts"
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    input_paths = [Path(path) for path in args.input_paths or ()]
    correct_fields = tuple(args.correct_fields or ("correct", "is_correct", "matches_reference"))
    id_filter_paths = tuple(Path(path) for path in args.id_filter_paths or ())
    summary = summarize_result_rows(
        input_paths=input_paths,
        correct_fields=correct_fields,
        limit=args.limit,
        id_filter_paths=id_filter_paths,
        confidence_level=args.confidence_level,
    )
    payload = summary.as_dict()
    payload.update(
        {
            "mode": "files",
            "input_paths": [str(path) for path in input_paths],
            "id_filter_paths": [str(path) for path in id_filter_paths],
            "correct_fields": list(correct_fields),
            "limit": args.limit,
        }
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
