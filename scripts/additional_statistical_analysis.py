"""Compute additional paired inference for the paper.

This script complements ``paper_revision_analysis.py``.  It distinguishes three
targets for the mean tool gain:

* item bootstrap: the 20 evaluated configurations are fixed;
* model bootstrap: the 133 benchmark items are fixed;
* crossed bootstrap: both configurations and items are resampled.

It also reports the model-sign test, multiplicity-adjusted per-model McNemar
tests, and paired comparisons among the three highest-scoring tool-enabled
configurations.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import paper_revision_analysis as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "generated"
SEED = 20260727
BOOTSTRAP_SAMPLES = 200_000
CROSSED_BOOTSTRAP_SAMPLES = 100_000


def outcome_matrix(
    data: dict[str, Any], item_ids: list[str], condition: str
) -> np.ndarray:
    matrix = np.empty((len(base.MODEL_LABELS), len(item_ids)), dtype=np.int8)
    for model_index, model_id in enumerate(base.MODEL_LABELS):
        by_id = {row["id"]: row for row in data[model_id][condition]}
        matrix[model_index] = [
            int(base.to_bool(by_id[item_id]["final_vote"])) for item_id in item_ids
        ]
    return matrix


def percentile_bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = values[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def crossed_bootstrap_ci(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int = CROSSED_BOOTSTRAP_SAMPLES,
    chunk_size: int = 2_000,
) -> tuple[float, float]:
    """Independently resample rows (models) and columns (items)."""

    model_count, item_count = differences.shape
    draws = np.empty(samples, dtype=float)
    offset = 0
    while offset < samples:
        chunk = min(chunk_size, samples - offset)
        model_weights = np.stack(
            [
                np.bincount(
                    rng.integers(0, model_count, size=model_count),
                    minlength=model_count,
                )
                for _ in range(chunk)
            ]
        )
        item_weights = np.stack(
            [
                np.bincount(
                    rng.integers(0, item_count, size=item_count),
                    minlength=item_count,
                )
                for _ in range(chunk)
            ]
        )
        draws[offset : offset + chunk] = (
            np.einsum(
                "bm,mi,bi->b",
                model_weights,
                differences,
                item_weights,
                optimize=True,
            )
            / (model_count * item_count)
        )
        offset += chunk
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def exact_sign_test(positives: int, total: int) -> tuple[float, float]:
    """Return one-sided (positive direction) and central two-sided p-values."""

    one_sided = sum(math.comb(total, k) for k in range(positives, total + 1)) / (
        2**total
    )
    smaller_tail_count = min(positives, total - positives)
    smaller_tail = sum(
        math.comb(total, k) for k in range(smaller_tail_count + 1)
    ) / (2**total)
    return one_sided, min(1.0, 2.0 * smaller_tail)


def paired_comparison(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str,
    second_name: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    difference = first.astype(float) - second.astype(float)
    first_only = int(np.sum((first == 1) & (second == 0)))
    second_only = int(np.sum((first == 0) & (second == 1)))
    ci_lo, ci_hi = percentile_bootstrap_ci(difference, rng=rng)
    return {
        "first_model": first_name,
        "second_model": second_name,
        "n": len(first),
        "first_correct": int(first.sum()),
        "second_correct": int(second.sum()),
        "difference": float(difference.mean()),
        "difference_ci_lo": ci_lo,
        "difference_ci_hi": ci_hi,
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "mcnemar_p_two_sided": base.exact_mcnemar_p(second_only, first_only),
    }


def format_percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}"


def format_p(value: float) -> str:
    return f"{value:.3g}"


def format_p_latex(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.001:
        coefficient, exponent = f"{value:.2e}".split("e")
        return rf"${coefficient}\mathbin{{\times}}10^{{{int(exponent)}}}$"
    return f"{value:.3f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_tables(
    path: Path,
    *,
    per_model_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Paired inference by configuration. Intervals resample the "
        r"133 paired items. Holm-adjusted values control family-wise error "
        r"across the 20 model-level McNemar tests.}",
        r"\label{tab:paired-model-inference}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Configuration & No tool (\%) & Agent (\%) & Gain [95\% CI] (pp) "
        r"& McNemar $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for row in sorted(per_model_rows, key=lambda item: item["tool_rate"], reverse=True):
        lines.append(
            f"{row['model']} & {format_percent(row['direct_rate'])} & "
            f"{format_percent(row['tool_rate'])} & "
            f"{format_percent(row['gain'])} "
            f"[{format_percent(row['gain_ci_lo'])}, "
            f"{format_percent(row['gain_ci_hi'])}] & "
            f"{format_p_latex(row['mcnemar_p'])} & "
            f"{format_p_latex(row['mcnemar_p_holm_20'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\caption{Paired comparisons among the three highest-scoring "
            r"tool-enabled configurations. A positive difference favors the "
            r"first configuration. Holm adjustment covers these three tests.}",
            r"\label{tab:top-group-paired}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Contrast & Difference [95\% CI] (pp) & McNemar $p$ & Holm $p$ \\",
            r"\midrule",
        ]
    )
    for row in top_rows:
        contrast = f"{row['first_model']}--{row['second_model']}"
        lines.append(
            f"{contrast} & {format_percent(row['difference'])} "
            f"[{format_percent(row['difference_ci_lo'])}, "
            f"{format_percent(row['difference_ci_hi'])}] & "
            f"{format_p_latex(row['mcnemar_p_two_sided'])} & "
            f"{format_p_latex(row['mcnemar_p_holm_3'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(
    path: Path,
    *,
    per_model_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    mean = summary["mean_gain"]
    lines = [
        "# Additional paired statistics",
        "",
        "All differences are computed on the same 133 items. Intervals are "
        "percentile bootstrap intervals with paired resampling.",
        "",
        "## Main results",
        "",
        f"- Mean gain: **{format_percent(mean)} percentage points**.",
        "- Fixed evaluated models, resampling items: "
        f"95% CI **{format_percent(summary['mean_gain_item_bootstrap_ci'][0])}"
        f" to {format_percent(summary['mean_gain_item_bootstrap_ci'][1])} pp**.",
        "- Fixed benchmark items, resampling models: "
        f"95% CI **{format_percent(summary['mean_gain_model_bootstrap_ci'][0])}"
        f" to {format_percent(summary['mean_gain_model_bootstrap_ci'][1])} pp**.",
        "- Resampling both models and items independently: "
        f"95% CI **{format_percent(summary['mean_gain_crossed_bootstrap_ci'][0])}"
        f" to {format_percent(summary['mean_gain_crossed_bootstrap_ci'][1])} pp**.",
        f"- Positive model-level point estimates: **{summary['positive_models']}/20**.",
        "- Exact sign test across model signs: "
        f"one-sided p = **{format_p(summary['model_sign_test_p_one_sided'])}**; "
        f"two-sided p = **{format_p(summary['model_sign_test_p_two_sided'])}**.",
        "- Model-level 95% paired-bootstrap intervals excluding zero: "
        f"**k = {summary['model_intervals_excluding_zero']}**.",
        "- Exact per-model McNemar tests below 0.05: "
        f"**{summary['per_model_mcnemar_p_below_0_05']} unadjusted**, "
        f"**{summary['per_model_mcnemar_holm_below_0_05']} after Holm correction "
        "over 20 models**.",
        "- Models retained by the 20-test Holm correction: "
        f"**{', '.join(summary['per_model_mcnemar_holm_significant_models'])}**.",
        "",
        "The sign test treats the 20 configuration signs as exchangeable units. "
        "Because related model configurations share architectures and all models "
        "see the same items, it is best presented as a descriptive robustness check, "
        "not as the primary inferential result.",
        "",
        "## Per-model paired gains",
        "",
        "| Model | No tool | Agent | Gain (pp) | 95% paired-bootstrap CI (pp) "
        "| Exact McNemar p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(per_model_rows, key=lambda item: item["tool_rate"], reverse=True):
        lines.append(
            f"| {row['model']} | {format_percent(row['direct_rate'])} | "
            f"{format_percent(row['tool_rate'])} | {format_percent(row['gain'])} | "
            f"[{format_percent(row['gain_ci_lo'])}, "
            f"{format_percent(row['gain_ci_hi'])}] | "
            f"{format_p(row['mcnemar_p'])} | "
            f"{format_p(row['mcnemar_p_holm_20'])} |"
        )

    lines.extend(
        [
            "",
            "## Tool-enabled comparisons within the top group",
            "",
            "A positive difference favors the first model.",
            "",
            "| Contrast | Correct | Difference (pp) | 95% paired-bootstrap CI (pp) "
            "| Discordant (first/second) | Exact McNemar p | Holm p (3 tests) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_rows:
        lines.append(
            f"| {row['first_model']} - {row['second_model']} | "
            f"{row['first_correct']}/{row['second_correct']} | "
            f"{format_percent(row['difference'])} | "
            f"[{format_percent(row['difference_ci_lo'])}, "
            f"{format_percent(row['difference_ci_hi'])}] | "
            f"{row['first_only_correct']}/{row['second_only_correct']} | "
            f"{format_p(row['mcnemar_p_two_sided'])} | "
            f"{format_p(row['mcnemar_p_holm_3'])} |"
        )

    lines.extend(
        [
            "",
            "None of the three top-group contrasts excludes zero. The ranking "
            "Sol > GPT-5.5 > Opus 4.8 is therefore descriptive on this benchmark; "
            "the paired data do not establish differences among them. Because this "
            "group was selected using the same observed scores, these are nominal, "
            "post-selection comparisons rather than confirmatory tests.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    with base.RESULTS_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)

    rows, item_ids = base.model_rows(data)
    direct = outcome_matrix(data, item_ids, "no_tool")
    tool = outcome_matrix(data, item_ids, "tool")
    differences = tool.astype(float) - direct.astype(float)
    rng = np.random.default_rng(SEED)

    per_model_p = [float(row["mcnemar_p"]) for row in rows]
    per_model_holm = holm_adjust(per_model_p)
    for row, adjusted_p in zip(rows, per_model_holm, strict=True):
        row["bootstrap_ci_excludes_zero"] = bool(
            row["gain_ci_lo"] > 0 or row["gain_ci_hi"] < 0
        )
        row["mcnemar_p_holm_20"] = adjusted_p

    model_gains = differences.mean(axis=1)
    item_gains = differences.mean(axis=0)
    mean_gain = float(differences.mean())
    model_ci = percentile_bootstrap_ci(model_gains, rng=rng)
    item_ci = percentile_bootstrap_ci(item_gains, rng=rng)
    crossed_ci = crossed_bootstrap_ci(differences, rng=rng)

    tool_by_model = {
        model_id: tool[index] for index, model_id in enumerate(base.MODEL_LABELS)
    }
    top_pairs = [
        ("gpt-5.6-sol", "gpt"),
        ("gpt-5.6-sol", "opus-4-8"),
        ("gpt", "opus-4-8"),
    ]
    top_rows = [
        paired_comparison(
            tool_by_model[first_id],
            tool_by_model[second_id],
            first_name=base.MODEL_LABELS[first_id],
            second_name=base.MODEL_LABELS[second_id],
            rng=rng,
        )
        for first_id, second_id in top_pairs
    ]
    top_holm = holm_adjust(
        [float(row["mcnemar_p_two_sided"]) for row in top_rows]
    )
    for row, adjusted_p in zip(top_rows, top_holm, strict=True):
        row["mcnemar_p_holm_3"] = adjusted_p

    positive_models = int(np.sum(model_gains > 0))
    zero_models = int(np.sum(model_gains == 0))
    if zero_models:
        raise ValueError("The 20-sign calculation assumes no zero gains")
    sign_one_sided, sign_two_sided = exact_sign_test(
        positive_models, len(base.MODEL_LABELS)
    )
    holm_significant_models = [
        row["model"]
        for row in rows
        if float(row["mcnemar_p_holm_20"]) < 0.05
    ]

    summary = {
        "models": len(base.MODEL_LABELS),
        "items": len(item_ids),
        "mean_gain": mean_gain,
        "mean_gain_item_bootstrap_ci": item_ci,
        "mean_gain_model_bootstrap_ci": model_ci,
        "mean_gain_crossed_bootstrap_ci": crossed_ci,
        "item_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "model_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "crossed_bootstrap_samples": CROSSED_BOOTSTRAP_SAMPLES,
        "positive_models": positive_models,
        "zero_gain_models": zero_models,
        "model_sign_test_p_one_sided": sign_one_sided,
        "model_sign_test_p_two_sided": sign_two_sided,
        "model_intervals_excluding_zero": int(
            sum(row["bootstrap_ci_excludes_zero"] for row in rows)
        ),
        "per_model_mcnemar_p_below_0_05": int(
            sum(value < 0.05 for value in per_model_p)
        ),
        "per_model_mcnemar_holm_below_0_05": int(
            sum(value < 0.05 for value in per_model_holm)
        ),
        "per_model_mcnemar_holm_significant_models": holm_significant_models,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_model_output = [
        {
            "model_id": row["model_id"],
            "model": row["model"],
            "n": row["n"],
            "direct_rate": row["direct_rate"],
            "tool_rate": row["tool_rate"],
            "gain": row["gain"],
            "gain_ci_lo": row["gain_ci_lo"],
            "gain_ci_hi": row["gain_ci_hi"],
            "bootstrap_ci_excludes_zero": row["bootstrap_ci_excludes_zero"],
            "direct_only": row["direct_only"],
            "tool_only": row["tool_only"],
            "mcnemar_p": row["mcnemar_p"],
            "mcnemar_p_holm_20": row["mcnemar_p_holm_20"],
        }
        for row in rows
    ]
    write_csv(OUTPUT_DIR / "per_model_inference.csv", per_model_output)
    write_csv(OUTPUT_DIR / "top_group_pairwise.csv", top_rows)
    write_latex_tables(
        OUTPUT_DIR / "additional_statistics_tables.tex",
        per_model_rows=rows,
        top_rows=top_rows,
    )
    with (OUTPUT_DIR / "additional_statistics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"summary": summary, "top_group_pairwise": top_rows},
            handle,
            indent=2,
        )
        handle.write("\n")
    write_markdown(
        OUTPUT_DIR / "additional_statistics.md",
        per_model_rows=rows,
        top_rows=top_rows,
        summary=summary,
    )

    print(json.dumps({"summary": summary, "top_group_pairwise": top_rows}, indent=2))


if __name__ == "__main__":
    main()
