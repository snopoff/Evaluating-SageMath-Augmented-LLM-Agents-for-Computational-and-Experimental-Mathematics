"""Score the probe responses and write the contamination report.

Reads ``probes.jsonl``, ``corpus.json`` and ``responses/*.jsonl`` and produces:

* ``scores.jsonl``          - one scored row per (model, probe).
* ``instant/<model>.json``  - instant-answer rows in the format expected by
                              ``judge_sympy_answers.py`` (symbolic check filled in; run the
                              judge on the file to add the LLM panel and re-run this report).
* ``report.md`` / ``report.json`` - group comparison per probe, guided-minus-unguided
                              deltas, quiz accuracy, and the link between memorisation
                              scores and direct (no-tool) correctness on the benchmark.

Usage::

    uv run python src/benchmark/contamination/report.py
    uv run python src/benchmark/contamination/report.py --results-map openai__gpt-5.5=gpt qwen__qwen3.7-max=qwen
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import rootutils

ROOT = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.benchmark.contamination.probes import load_probes  # noqa: E402
from src.benchmark.contamination.scoring import (  # noqa: E402
    looks_like_refusal,
    parse_instant_answer,
    parse_quiz_letter,
    score_continuation,
)
from src.benchmark.judge_sympy_answers import load_results  # noqa: E402
from src.benchmark.sympy_compare import SympyAnswerComparator  # noqa: E402

DEFAULT_DIR = ROOT / "data/reviews/contamination"
DEFAULT_RESULTS_ROOT = ROOT / "data/results"
GROUP_ORDER = ("famous", "benchmark", "same_vintage", "fresh")
CONTINUATION_PROBES = ("abstract", "pretext", "proof")
EXTRACTABLE_RUN = 50
QUIZ_CHANCE = 0.2

# Response label -> results folder under data/results (direct correctness of the same model).
DEFAULT_RESULTS_MAP = {
    "openai__gpt-5.5": "gpt",
    "anthropic__claude-opus-4-8": "opus-4-8",
    "openrouter__deepseek_deepseek-v4-pro": "deepseek-v4-pro",
}


# --------------------------------------------------------------------------- statistics helpers


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def bootstrap_ci(clusters: dict[str, list[float]], *, iterations: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95% CI of the mean with resampling over clusters (papers), not rows."""

    keys = list(clusters)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample: list[float] = []
        for _ in keys:
            sample.extend(clusters[rng.choice(keys)])
        means.append(statistics.fmean(sample))
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """P(score_pos > score_neg) with ties counted half: the Mann-Whitney AUROC."""

    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for n in negative:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positive) * len(negative))


def binomial_p_at_least(successes: int, trials: int, p: float) -> float:
    return sum(math.comb(trials, k) * p**k * (1 - p) ** (trials - k) for k in range(successes, trials + 1))


def fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_ci(ci: tuple[float, float] | None) -> str:
    return "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"


# --------------------------------------------------------------------------- loading and scoring


def load_responses(responses_dir: Path, prompt_version: int) -> dict[str, dict[str, dict[str, Any]]]:
    responses: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(responses_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Retried probes append a second row; keep a successful one over an error.
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("prompt_version", 1) != prompt_version:
                continue
            if row["probe_id"] not in by_id or by_id[row["probe_id"]].get("error"):
                by_id[row["probe_id"]] = row
        responses[path.stem] = by_id
    return responses


def score_rows(probes: list[dict[str, Any]], responses: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    comparator = SympyAnswerComparator()
    rows: list[dict[str, Any]] = []
    for model, by_id in responses.items():
        for probe in probes:
            response = by_id.get(probe["probe_id"])
            if response is None:
                continue
            text = response.get("response") or ""
            row: dict[str, Any] = {
                "model": model,
                "probe_id": probe["probe_id"],
                "probe": probe["probe"],
                "mode": probe["mode"],
                "group": probe["group"],
                "arxiv_id": probe["arxiv_id"],
                "cohort": probe.get("cohort"),
                "problem_id": probe.get("problem_id"),
                "error": response.get("error"),
                "response_tokens": response.get("response_tokens", 0),
                "refusal": looks_like_refusal(text),
            }
            if probe["probe"] in CONTINUATION_PROBES:
                row.update(score_continuation(text, probe["target"]).as_dict())
            elif probe["probe"] in ("quiz", "quiz_abstract"):
                letter = parse_quiz_letter(text)
                row["letter"] = letter
                row["correct"] = letter == probe["target"]
            elif probe["probe"] == "instant":
                parsed = parse_instant_answer(text)
                reference = probe["meta"].get("ground_truth_sympy_answer") or ""
                try:
                    result = comparator.score(comparator.coerce(parsed["sympy_answer"]), comparator.coerce(reference))
                    sympy_correct, match_type = bool(result.correct), result.match_type
                except Exception as exc:  # noqa: BLE001 - unparsable answers count as wrong
                    sympy_correct, match_type = False, f"error: {exc.__class__.__name__}"
                declined = parsed["final_answer"].strip().upper() in {"NONE", "", "N/A"}
                row.update(
                    {
                        "final_answer": parsed["final_answer"],
                        "sympy_answer": parsed["sympy_answer"],
                        "parsed": parsed["parsed"],
                        "sympy_correct": sympy_correct and not declined,
                        "match_type": match_type,
                        "declined": declined,
                    }
                )
            rows.append(row)
    return rows


def write_instant_files(rows: list[dict[str, Any]], probes_by_id: dict[str, dict[str, Any]], instant_dir: Path) -> None:
    """Judge-compatible files: run judge_sympy_answers.py --input <file> to add the panel."""

    instant_dir.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["probe"] != "instant":
            continue
        probe = probes_by_id[row["probe_id"]]
        by_model[row["model"]].append(
            {
                "id": row["problem_id"],
                "question": probe["user"].split("Question:\n\n", 1)[-1].split("\n\nReply with a JSON", 1)[0],
                "ground_truth": probe["meta"].get("ground_truth"),
                "ground_truth_sympy_answer": probe["meta"].get("ground_truth_sympy_answer"),
                "model_final_answer": row["final_answer"],
                "model_sympy_answer": row["sympy_answer"],
                "correct": row["sympy_correct"],
                "match_type": row["match_type"],
                "arxiv_id": row["arxiv_id"],
            }
        )
    for model, model_rows in by_model.items():
        (instant_dir / f"{model}.json").write_text(json.dumps(model_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_instant_judgements(rows: list[dict[str, Any]], instant_dir: Path) -> dict[str, str]:
    """Upgrade instant correctness to sympy OR judge-majority where a judged file exists."""

    sources: dict[str, str] = {}
    for model in {row["model"] for row in rows if row["probe"] == "instant"}:
        judged_path = instant_dir / f"{model}_sympy_judge_verdicts.json"
        if not judged_path.is_file():
            sources[model] = "symbolic check only"
            continue
        judged = {str(r["id"]): r for r in load_results(judged_path)}
        for row in rows:
            if row["model"] == model and row["probe"] == "instant":
                verdict = judged.get(row["problem_id"])
                row["judge_correct"] = bool(verdict.get("majority_vote_correctness")) if verdict else None
        sources[model] = f"symbolic check OR judge majority ({judged_path.name})"
    for row in rows:
        if row["probe"] == "instant":
            row["correct"] = bool((row["sympy_correct"] or row.get("judge_correct")) and not row["declined"])
    return sources


# --------------------------------------------------------------------------- direct correctness of the benchmark runs


def load_direct_correctness(results_root: Path, folder: str, mode: str = "no_tool") -> dict[str, bool] | None:
    frozen = results_root / "paper_results.json"
    if frozen.is_file():
        data = json.loads(frozen.read_text())
        if folder in data:
            return {r["id"]: r["final_vote"] is True for r in data[folder][mode]}
    output_dir = results_root / folder / mode / "output"
    if not (output_dir / "full.json").is_file():
        return None
    correct = {str(r["id"]): r.get("correct") is True for r in load_results(output_dir / "full.json")}
    if (output_dir / "judge_verdicts.json").is_file():
        for r in load_results(output_dir / "judge_verdicts.json"):
            if r.get("majority_vote_correctness"):
                correct[str(r["id"])] = True
    return correct


def guess_results_folder(label: str, results_root: Path, explicit: dict[str, str]) -> str | None:
    if label in explicit:
        return explicit[label]
    if label in DEFAULT_RESULTS_MAP:
        return DEFAULT_RESULTS_MAP[label]
    if label.startswith("qwen__") or label.startswith("openrouter__qwen"):
        return "qwen"
    if label.startswith("google__gemini-3.1"):
        return "gemini-3.1"
    if label.startswith("google__gemini-3.5"):
        return "gemini"
    return None


# --------------------------------------------------------------------------- report sections


def section_groups(rows: list[dict[str, Any]], out: list[str], payload: dict[str, Any]) -> None:
    out.append("## 1. Verbatim-recall probes by group\n")
    out.append(
        "ROUGE-L between the model continuation and the true text; `8-gram` = share of the model's 8-grams found in "
        f"the target; `extractable` = share of papers with a verbatim run of >= {EXTRACTABLE_RUN} tokens; "
        "CI = 95% paper-level bootstrap of the mean ROUGE-L. AUROC compares the benchmark papers with a control "
        "group (0.5 = indistinguishable, 1.0 = every benchmark paper recalled better than every control paper).\n"
    )
    payload["groups"] = []
    models = sorted({row["model"] for row in rows})
    for probe in CONTINUATION_PROBES:
        for mode in ("guided", "unguided"):
            subset = [r for r in rows if r["probe"] == probe and r["mode"] == mode and not r["error"]]
            if not subset:
                continue
            out.append(f"### {probe} ({mode})\n")
            out.append("| model | group | n | ROUGE-L mean [CI] | median | 8-gram | extractable | refusals |")
            out.append("|---|---|---:|---|---:|---:|---:|---:|")
            for model in models:
                group_scores: dict[str, list[float]] = {}
                for group in GROUP_ORDER:
                    members = [r for r in subset if r["model"] == model and r["group"] == group]
                    if not members:
                        continue
                    scores = [r["rouge_l"] for r in members]
                    group_scores[group] = scores
                    clusters: dict[str, list[float]] = defaultdict(list)
                    for r in members:
                        clusters[r["arxiv_id"]].append(r["rouge_l"])
                    ci = bootstrap_ci(clusters)
                    entry = {
                        "probe": probe,
                        "mode": mode,
                        "model": model,
                        "group": group,
                        "n": len(members),
                        "rouge_l_mean": mean(scores),
                        "rouge_l_ci": ci,
                        "rouge_l_median": statistics.median(scores),
                        "ngram8_mean": mean([r["ngram8_hit"] for r in members]),
                        "extractable_share": mean([1.0 if r["longest_run"] >= EXTRACTABLE_RUN else 0.0 for r in members]),
                        "refusal_share": mean([1.0 if r["refusal"] else 0.0 for r in members]),
                    }
                    payload["groups"].append(entry)
                    out.append(
                        f"| {model} | {group} | {entry['n']} | {fmt(entry['rouge_l_mean'])}{fmt_ci(ci)} | "
                        f"{fmt(entry['rouge_l_median'])} | {fmt(entry['ngram8_mean'])} | "
                        f"{fmt(entry['extractable_share'], 2)} | {fmt(entry['refusal_share'], 2)} |"
                    )
                if "benchmark" in group_scores:
                    for control in ("same_vintage", "fresh"):
                        if control in group_scores:
                            value = auroc(group_scores["benchmark"], group_scores[control])
                            payload["groups"].append({"probe": probe, "mode": mode, "model": model, "auroc_vs": control, "auroc": value})
                            out.append(f"| {model} | AUROC benchmark vs {control} | | {fmt(value)} | | | | |")
            out.append("")


def section_guidance_delta(rows: list[dict[str, Any]], out: list[str], payload: dict[str, Any]) -> None:
    out.append("## 2. Guided minus unguided\n")
    out.append(
        "Naming the paper should only help a model that has stored this specific document; generic fluency "
        "is the same in both modes. Values are mean paired differences in ROUGE-L with a paper-level bootstrap CI.\n"
    )
    out.append("| model | probe | group | n | delta ROUGE-L [CI] |")
    out.append("|---|---|---|---:|---|")
    payload["guidance_delta"] = []
    for probe in ("abstract", "pretext"):
        guided = {(r["model"], r["arxiv_id"]): r for r in rows if r["probe"] == probe and r["mode"] == "guided" and not r["error"]}
        unguided = {(r["model"], r["arxiv_id"]): r for r in rows if r["probe"] == probe and r["mode"] == "unguided" and not r["error"]}
        if not guided or not unguided:
            continue
        for model in sorted({key[0] for key in guided}):
            for group in GROUP_ORDER:
                clusters: dict[str, list[float]] = {}
                for (m, arxiv_id), g in guided.items():
                    u = unguided.get((m, arxiv_id))
                    if m == model and g["group"] == group and u is not None:
                        clusters[arxiv_id] = [g["rouge_l"] - u["rouge_l"]]
                if not clusters:
                    continue
                deltas = [v[0] for v in clusters.values()]
                ci = bootstrap_ci(clusters)
                payload["guidance_delta"].append({"probe": probe, "model": model, "group": group, "n": len(deltas), "delta_mean": mean(deltas), "ci": ci})
                out.append(f"| {model} | {probe} | {group} | {len(deltas)} | {fmt(mean(deltas))}{fmt_ci(ci)} |")
    out.append("")


def section_quiz(rows: list[dict[str, Any]], out: list[str], payload: dict[str, Any]) -> None:
    subset = [r for r in rows if r["probe"] in ("quiz", "quiz_abstract") and not r["error"]]
    if not subset:
        return
    out.append("## 3. Recognition quizzes\n")
    out.append(
        f"Five options, chance = {QUIZ_CHANCE:.0%}; one-sided exact binomial p-value against chance. "
        "`quiz` = RealMath question wording (benchmark leakage). `quiz_abstract` = the same perturbation "
        "scheme on paper abstracts, per group: the fresh group is the control, because the model cannot "
        "have seen those abstracts, so its accuracy there is what fluency alone buys. Read `quiz` against "
        "it: only the excess over the fresh-abstract accuracy is evidence of exposure.\n"
    )
    out.append("| model | probe | group | n | accuracy | p vs chance | answered E | letters |")
    out.append("|---|---|---|---:|---:|---:|---:|---|")
    payload["quiz"] = []
    for model in sorted({r["model"] for r in subset}):
        for probe, group in [("quiz", "benchmark")] + [("quiz_abstract", g) for g in GROUP_ORDER]:
            members = [r for r in subset if r["model"] == model and r["probe"] == probe and r["group"] == group]
            if not members:
                continue
            correct = sum(1 for r in members if r["correct"])
            p_value = binomial_p_at_least(correct, len(members), QUIZ_CHANCE)
            letters: dict[str, int] = defaultdict(int)
            for r in members:
                if r["letter"] is not None:
                    letters[r["letter"]] += 1
            entry = {
                "model": model,
                "probe": probe,
                "group": group,
                "n": len(members),
                "accuracy": correct / len(members),
                "p_value": p_value,
                "e_share": letters["E"] / len(members),
            }
            payload["quiz"].append(entry)
            out.append(
                f"| {model} | {probe} | {group} | {len(members)} | {entry['accuracy']:.3f} | {p_value:.3g} | "
                f"{entry['e_share']:.2f} | " + ", ".join(f"{k}:{v}" for k, v in sorted(letters.items())) + " |"
            )
    out.append("")


def section_instant(rows: list[dict[str, Any]], direct: dict[str, dict[str, bool]], sources: dict[str, str], out: list[str], payload: dict[str, Any]) -> None:
    subset = [r for r in rows if r["probe"] == "instant" and not r["error"]]
    if not subset:
        return
    out.append("## 4. Instant answers (no derivation allowed)\n")
    out.append(
        "The question with an instruction to answer immediately. A research-level result that is produced correctly "
        "without any derivation is either trivial or recalled. `direct` is the model's regular no-tool accuracy on the "
        "same items; `instant & direct` counts items solved both ways. Correctness source per model is listed below.\n"
    )
    out.append("| model | n | instant correct | declined (NONE) | direct correct | instant & direct | instant-correct among direct-wrong |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    payload["instant"] = []
    for model in sorted({r["model"] for r in subset}):
        members = [r for r in subset if r["model"] == model]
        n = len(members)
        instant_correct = sum(1 for r in members if r["correct"])
        declined = sum(1 for r in members if r["declined"])
        entry: dict[str, Any] = {"model": model, "n": n, "instant_correct": instant_correct, "declined": declined, "source": sources.get(model)}
        direct_map = direct.get(model)
        if direct_map is not None:
            both = sum(1 for r in members if r["correct"] and direct_map.get(r["problem_id"]))
            direct_correct = sum(1 for r in members if direct_map.get(r["problem_id"]))
            instant_only = sum(1 for r in members if r["correct"] and not direct_map.get(r["problem_id"]))
            entry.update({"direct_correct": direct_correct, "both": both, "instant_only": instant_only})
            out.append(f"| {model} | {n} | {instant_correct} ({instant_correct / n:.1%}) | {declined} | {direct_correct} ({direct_correct / n:.1%}) | {both} | {instant_only} |")
        else:
            out.append(f"| {model} | {n} | {instant_correct} ({instant_correct / n:.1%}) | {declined} | - | - | - |")
        payload["instant"].append(entry)
    out.append("")
    for model, source in sorted(sources.items()):
        out.append(f"- {model}: {source}")
    out.append("")


def _item_memorization(rows: list[dict[str, Any]], model: str) -> dict[str, dict[str, float]]:
    """Per benchmark paper: paper-level recall scores for one model."""

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["model"] != model or r["group"] != "benchmark" or r["probe"] not in CONTINUATION_PROBES or r["error"]:
            continue
        scores[r["arxiv_id"]][f"{r['probe']}_{r['mode']}"] = r["rouge_l"]
    return scores


def section_strata(
    rows: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    direct: dict[str, dict[str, bool]],
    answer_types: dict[str, str],
    out: list[str],
    payload: dict[str, Any],
) -> None:
    models = sorted(direct)
    if not models:
        out.append("## 5. Memorisation vs. direct correctness\n\nNo results folders matched the response labels; pass `--results-map`.\n")
        return
    problem_paper = {p["problem_id"]: p["arxiv_id"] for p in probes if p.get("problem_id")}
    problem_cohort = {p["problem_id"]: p.get("cohort") for p in probes if p.get("problem_id")}
    problems = sorted(problem_paper)
    solved_by = {pid: sum(1 for m in models if direct[m].get(pid)) for pid in problems}
    instant_by = {(r["model"], r["problem_id"]): r["correct"] for r in rows if r["probe"] == "instant" and not r["error"]}

    out.append("## 5. Memorisation vs. direct correctness on the benchmark\n")
    out.append(
        "If contamination drove the direct baseline, items that the models solve should come from papers they recall "
        "better. AUROC = P(recall score of a directly-solved item > score of an unsolved item), per model, on the "
        "paper-level ROUGE-L of each probe; 0.5 means recall does not separate solved from unsolved items.\n"
    )
    out.append("| model | probe | solved n | unsolved n | mean solved | mean unsolved | AUROC |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    payload["item_auroc"] = []
    for model in models:
        memo = _item_memorization(rows, model)
        keys = sorted({k for scores in memo.values() for k in scores})
        for key in keys:
            solved = [memo[problem_paper[pid]][key] for pid in problems if direct[model].get(pid) and key in memo.get(problem_paper[pid], {})]
            unsolved = [memo[problem_paper[pid]][key] for pid in problems if not direct[model].get(pid) and key in memo.get(problem_paper[pid], {})]
            value = auroc(solved, unsolved)
            payload["item_auroc"].append({"model": model, "probe": key, "solved_n": len(solved), "unsolved_n": len(unsolved), "auroc": value})
            out.append(f"| {model} | {key} | {len(solved)} | {len(unsolved)} | {fmt(mean(solved))} | {fmt(mean(unsolved))} | {fmt(value)} |")
    out.append("")

    out.append("### Strata\n")
    out.append(
        f"`solved by k` counts the {len(models)} probed models' no-tool runs. Recall scores are averaged over the "
        "probed models and over the items in the stratum; `instant` is the share of (model, item) pairs answered "
        "correctly with no derivation.\n"
    )
    keys = sorted({k for m in models for scores in _item_memorization(rows, m).values() for k in scores})
    out.append("| stratum | items | " + " | ".join(keys) + " | instant |")
    out.append("|---|---:|" + "---:|" * (len(keys) + 1))
    payload["strata"] = []
    memo_by_model = {m: _item_memorization(rows, m) for m in models}

    def stratum_row(name: str, members: list[str]) -> None:
        if not members:
            return
        entry: dict[str, Any] = {"stratum": name, "items": len(members)}
        cells: list[str] = []
        for key in keys:
            values = [memo_by_model[m][problem_paper[pid]][key] for m in models for pid in members if key in memo_by_model[m].get(problem_paper[pid], {})]
            entry[key] = mean(values)
            cells.append(fmt(mean(values)))
        instant_values = [1.0 if instant_by.get((m, pid)) else 0.0 for m in models for pid in members if (m, pid) in instant_by]
        entry["instant"] = mean(instant_values)
        payload["strata"].append(entry)
        out.append(f"| {name} | {len(members)} | " + " | ".join(cells) + f" | {fmt(mean(instant_values), 2)} |")

    stratum_row("all", problems)
    for answer_type in sorted(set(answer_types.values())):
        stratum_row(f"answer type: {answer_type}", [pid for pid in problems if answer_types.get(pid) == answer_type])
    for k in range(len(models) + 1):
        stratum_row(f"solved by {k}/{len(models)} models", [pid for pid in problems if solved_by[pid] == k])
    for cohort in ("2022", "2024"):
        stratum_row(f"cohort {cohort}", [pid for pid in problems if problem_cohort.get(pid) == cohort])
    out.append("")


# --------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-scores", action="store_true", help="Use immutable saved scores without rescoring or judging.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/generated/contamination")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--results-map", nargs="*", default=[], help="label=folder pairs linking responses to data/results runs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    probes = load_probes(args.dir / "probes.jsonl")
    probes_by_id = {p["probe_id"]: p for p in probes}
    prompt_version = max(p.get("prompt_version", 1) for p in probes)
    responses = load_responses(args.dir / "responses", prompt_version)
    if not responses:
        print(f"no responses under {args.dir / 'responses'}")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_scores:
        rows = [json.loads(line) for line in (args.dir / "scores.jsonl").read_text().splitlines() if line.strip()]
        sources = {model: "saved scores" for model in responses}
    else:
        rows = score_rows(probes, responses)
        write_instant_files(rows, probes_by_id, args.output_dir / "instant")
        sources = apply_instant_judgements(rows, args.dir / "instant")
        with (args.output_dir / "scores.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    explicit = dict(item.split("=", 1) for item in args.results_map)
    direct: dict[str, dict[str, bool]] = {}
    for label in responses:
        folder = guess_results_folder(label, args.results_root, explicit)
        correctness = load_direct_correctness(args.results_root, folder) if folder else None
        if correctness is not None:
            direct[label] = correctness
    answer_types: dict[str, str] = {}
    for folder in set(DEFAULT_RESULTS_MAP.values()) | set(explicit.values()):
        full = args.results_root / folder / "no_tool" / "output" / "full.json"
        if full.is_file():
            answer_types = {str(r["id"]): str(r.get("answer_type")) for r in load_results(full)}
            break

    frozen = args.results_root / "paper_results.json"
    if frozen.is_file():
        data = json.loads(frozen.read_text())
        answer_types = {r["id"]: r["answer_type"] for r in data["gpt"]["no_tool"]}
    out: list[str] = ["# Contamination probes: report\n", "Offline replay: saved probe scores; benchmark correlations use frozen headline decisions.\n"]
    corpus = json.loads((args.dir / "corpus.json").read_text(encoding="utf-8"))
    out.append(f"Corpus built {corpus.get('built_at')}; groups: " + ", ".join(f"{g}={s['papers']}" for g, s in corpus["summary"].items()) + ".")
    out.append(f"Models: {', '.join(sorted(responses))}. Errors: " + ", ".join(f"{m}={sum(1 for r in rows if r['model'] == m and r['error'])}" for m in sorted(responses)) + ".\n")
    abstract_fallbacks = [p for p in probes if p["probe"] == "abstract" and p["mode"] == "guided" and p.get("meta", {}).get("split_method") == "word_fallback"]
    if abstract_fallbacks:
        out.append(
            "Abstract-continuation coverage: existing sentence-boundary splits are retained; "
            "short or single-sentence abstracts use the word boundary nearest 40% of the text. "
            f"This fallback retains {len(abstract_fallbacks)} previously excluded papers "
            "without changing their source texts or the existing eligible prompts.\n"
        )
    payload: dict[str, Any] = {}
    section_groups(rows, out, payload)
    section_guidance_delta(rows, out, payload)
    section_quiz(rows, out, payload)
    section_instant(rows, direct, sources, out, payload)
    section_strata(rows, probes, direct, answer_types, out, payload)
    out.append(
        "## Caveats\n\n"
        "- Black-box probes detect verbatim recall; a model can be trained on a paper and recall nothing verbatim, "
        "so low scores do not prove absence of exposure. High recall of a paper also does not prove the benchmark "
        "answer was memorised; section 5 is the relevant link.\n"
        "- The fresh group is only a floor if every probed model's training cutoff precedes the fresh window; check "
        "the cutoffs before quoting it.\n"
        "- The famous group is a ceiling reached through millions of copies; ordinary papers are not expected to get there.\n"
    )
    (args.output_dir / "report.md").write_text("\n".join(out), encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(out))
    print(f"\nwrote {args.output_dir / 'report.md'} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
