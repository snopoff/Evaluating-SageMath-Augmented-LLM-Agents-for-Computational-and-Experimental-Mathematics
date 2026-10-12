import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NORMALIZED_PATH = ROOT / "data/processed/normalized_realmath.json"
DEFAULT_TAGS_PATH = ROOT / "data/construction/arxiv_tags_by_problem.json"
DEFAULT_USED_PATH = ROOT / "data/processed/problems_to_run.json"
DEFAULT_OUTPUT_PATH = ROOT / "outputs/unused_normalized_problems_target_primary_categories.json"
DEFAULT_STATS_PATH = ROOT / "outputs/unused_normalized_problems_target_primary_categories_stats.json"
DEFAULT_TARGET_CATEGORIES = (
    "math.AG",
    "math.GT",
    "math.AP",
    "math.GR",
    "math.PR",
    "math.AT",
    "math.CA",
    "math.RA",
    "math.AC",
)


def sample_unused_problems_by_arxiv_category(
    *,
    normalized_path: str | Path = DEFAULT_NORMALIZED_PATH,
    tags_path: str | Path = DEFAULT_TAGS_PATH,
    used_path: str | Path = DEFAULT_USED_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    stats_path: str | Path = DEFAULT_STATS_PATH,
    target_categories: Sequence[str] = DEFAULT_TARGET_CATEGORIES,
    max_per_category: int | None = None,
    target_count_per_category: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    include_arxiv_metadata: bool = True,
) -> tuple[Path, Path]:
    normalized_rows = _load_rows(Path(normalized_path))
    tags_by_id = _load_tags_by_id(Path(tags_path))
    used_ids = _load_ids(Path(used_path))
    target_category_set = set(target_categories)

    selected_by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in target_categories}
    eligible_by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in target_categories}
    used_counts_by_category = _used_primary_category_counts(used_ids, tags_by_id, target_category_set)
    missing_tag_ids: list[str] = []

    for row in normalized_rows:
        problem_id = _required_str(row, "id")
        tag_row = tags_by_id.get(problem_id)
        if tag_row is None:
            missing_tag_ids.append(problem_id)
            continue

        categories = _required_categories(tag_row)
        top_category = categories[0]
        if top_category not in target_category_set or problem_id in used_ids:
            continue

        candidate = dict(row)
        if include_arxiv_metadata:
            candidate.update(
                {
                    "arxiv_id": tag_row.get("arxiv_id"),
                    "arxiv_link": tag_row.get("link"),
                    "arxiv_categories": categories,
                    "arxiv_top_category": top_category,
                }
            )
        eligible_by_category[top_category].append(candidate)

    rng = random.Random(seed)
    for category in target_categories:
        candidates = list(eligible_by_category[category])
        if shuffle:
            rng.shuffle(candidates)

        category_limit = _category_limit(
            category=category,
            available_count=len(candidates),
            used_count=used_counts_by_category.get(category, 0),
            max_per_category=max_per_category,
            target_count_per_category=target_count_per_category,
        )
        selected_by_category[category] = candidates[:category_limit]

    selected_rows = [row for category in target_categories for row in selected_by_category[category]]
    if shuffle:
        rng.shuffle(selected_rows)

    output_destination = Path(output_path)
    output_destination.parent.mkdir(parents=True, exist_ok=True)
    with output_destination.open("w", encoding="utf-8") as handle:
        json.dump(selected_rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    stats = _build_stats(
        normalized_rows=normalized_rows,
        used_ids=used_ids,
        target_categories=target_categories,
        eligible_by_category=eligible_by_category,
        selected_by_category=selected_by_category,
        used_counts_by_category=used_counts_by_category,
        missing_tag_ids=missing_tag_ids,
        max_per_category=max_per_category,
        target_count_per_category=target_count_per_category,
        shuffle=shuffle,
        seed=seed,
    )
    stats_destination = Path(stats_path)
    stats_destination.parent.mkdir(parents=True, exist_ok=True)
    with stats_destination.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return output_destination, stats_destination


def _category_limit(
    *,
    category: str,
    available_count: int,
    used_count: int,
    max_per_category: int | None,
    target_count_per_category: int | None,
) -> int:
    limit = available_count
    if target_count_per_category is not None:
        limit = min(limit, max(target_count_per_category - used_count, 0))
    if max_per_category is not None:
        limit = min(limit, max_per_category)
    if limit < 0:
        raise ValueError(f"Computed negative sample limit for {category}")
    return limit


def _used_primary_category_counts(
    used_ids: set[str],
    tags_by_id: dict[str, dict[str, Any]],
    target_categories: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for problem_id in used_ids:
        tag_row = tags_by_id.get(problem_id)
        if tag_row is None:
            continue
        top_category = _required_categories(tag_row)[0]
        if top_category in target_categories:
            counts[top_category] += 1
    return counts


def _build_stats(
    *,
    normalized_rows: list[dict[str, Any]],
    used_ids: set[str],
    target_categories: Sequence[str],
    eligible_by_category: dict[str, list[dict[str, Any]]],
    selected_by_category: dict[str, list[dict[str, Any]]],
    used_counts_by_category: Counter[str],
    missing_tag_ids: list[str],
    max_per_category: int | None,
    target_count_per_category: int | None,
    shuffle: bool,
    seed: int,
) -> dict[str, Any]:
    selected_count = sum(len(rows) for rows in selected_by_category.values())
    eligible_count = sum(len(rows) for rows in eligible_by_category.values())
    return {
        "normalized_problem_count": len(normalized_rows),
        "used_problem_count": len(used_ids),
        "target_categories": list(target_categories),
        "target_count_per_category": target_count_per_category,
        "max_per_category": max_per_category,
        "shuffle": shuffle,
        "seed": seed,
        "missing_tag_count": len(missing_tag_ids),
        "missing_tag_ids": missing_tag_ids,
        "eligible_unused_problem_count": eligible_count,
        "selected_problem_count": selected_count,
        "counts_by_category": [
            {
                "category": category,
                "used_training_count": used_counts_by_category.get(category, 0),
                "eligible_unused_count": len(eligible_by_category[category]),
                "selected_count": len(selected_by_category[category]),
                "selected_ids": [_required_str(row, "id") for row in selected_by_category[category]],
            }
            for category in target_categories
        ],
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [row for row in payload["results"] if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("problems"), list):
        return [row for row in payload["problems"] if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON payload in {path}")


def _load_tags_by_id(path: Path) -> dict[str, dict[str, Any]]:
    tags_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in _load_rows(path):
        problem_id = _required_str(row, "id")
        if problem_id in tags_by_id:
            duplicate_ids.add(problem_id)
        tags_by_id[problem_id] = row
    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:10])
        suffix = "" if len(duplicate_ids) <= 10 else f", ... ({len(duplicate_ids)} total)"
        raise ValueError(f"Duplicate tag rows for problem IDs: {preview}{suffix}")
    return tags_by_id


def _load_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in _load_rows(path):
        value = row.get("id")
        if isinstance(value, str) and value:
            ids.add(value)
    if not ids:
        raise ValueError(f"No problem IDs found in {path}")
    return ids


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string field {key!r}.")
    return value


def _required_categories(row: dict[str, Any]) -> list[str]:
    categories = row.get("categories") or row.get("arxiv_categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"Expected non-empty categories list for problem {row.get('id')!r}.")
    if not all(isinstance(category, str) and category for category in categories):
        raise ValueError(f"Expected every category to be a non-empty string for problem {row.get('id')!r}.")
    return list(categories)


def _parse_categories(value: str) -> list[str]:
    categories = [category.strip() for category in value.split(",") if category.strip()]
    if not categories:
        raise argparse.ArgumentTypeError("At least one category is required.")
    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample unused normalized problems whose primary arXiv category is in a target list.")
    parser.add_argument("--normalized-path", default=str(DEFAULT_NORMALIZED_PATH), help="Path to normalized_problems.json.")
    parser.add_argument("--tags-path", default=str(DEFAULT_TAGS_PATH), help="Path to arxiv_tags_by_problem.json.")
    parser.add_argument("--used-path", default=str(DEFAULT_USED_PATH), help="Result JSON containing already-used problem IDs.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Destination JSON for selected normalized problems.")
    parser.add_argument("--stats-path", default=str(DEFAULT_STATS_PATH), help="Destination JSON for selection statistics.")
    parser.add_argument(
        "--categories",
        type=_parse_categories,
        default=list(DEFAULT_TARGET_CATEGORIES),
        help="Comma-separated primary arXiv categories to keep.",
    )
    parser.add_argument("--max-per-category", type=int, default=None, help="Optional cap on selected rows per category.")
    parser.add_argument(
        "--target-count-per-category",
        type=int,
        default=None,
        help="Optional final training target per category; selects at most target minus already-used rows.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle candidates before selecting.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when --shuffle is set.")
    parser.add_argument(
        "--preserve-schema",
        action="store_true",
        help="Do not add arXiv metadata fields to output rows.",
    )
    args = parser.parse_args()

    output_path, stats_path = sample_unused_problems_by_arxiv_category(
        normalized_path=args.normalized_path,
        tags_path=args.tags_path,
        used_path=args.used_path,
        output_path=args.output_path,
        stats_path=args.stats_path,
        target_categories=args.categories,
        max_per_category=args.max_per_category,
        target_count_per_category=args.target_count_per_category,
        shuffle=args.shuffle,
        seed=args.seed,
        include_arxiv_metadata=not args.preserve_schema,
    )
    print(json.dumps({"output_path": str(output_path), "stats_path": str(stats_path)}, indent=2))


if __name__ == "__main__":
    main()
