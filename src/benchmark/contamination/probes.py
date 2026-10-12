"""Turn the corpus into prompt/target records for the contamination probes.

Probes (``probe`` field of every record):

* ``abstract``  - first ~40% of the abstract, continue the text.  Run ``guided`` (with
                  arXiv id and title) and ``unguided``; abstracts are the most copied part
                  of a paper, so this is the upper bound on exposure to the paper as such.
* ``pretext``   - first half of the paragraph right before the benchmark theorem,
                  continue verbatim.  Run twice: ``guided`` (with arXiv id and title)
                  and ``unguided`` (no metadata).  Their difference isolates recall of
                  this document from generic mathematical fluency.
* ``proof``     - full theorem statement, write the opening of the proof as in the
                  paper (guided).  Text that cannot be derived from the mathematics.
* ``instant``   - the benchmark question, final answer only, no derivation allowed
                  (benchmark problems only).  A research-level formula cannot be derived
                  in a few tokens; it can only be recalled.
* ``quiz``      - Data Contamination Quiz on the RealMath question wording: the
                  original plus three deterministic perturbations and "none", chance is
                  20% (benchmark problems only).  Tests leakage of the released benchmark
                  rather than of the source paper.
* ``quiz_abstract`` - the same quiz on paper abstracts, for every group.  This is the
                  control for ``quiz``: on fresh papers the model cannot recognise the
                  original, so any accuracy above chance there measures how detectable the
                  perturbations are by fluency alone rather than memory.

Usage::

    uv run python src/benchmark/contamination/probes.py
    uv run python src/benchmark/contamination/probes.py --probes abstract instant --show 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import rootutils

ROOT = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

DEFAULT_DIR = ROOT / "data/reviews/contamination"
DEFAULT_PROBLEMS = ROOT / "data/processed/problems_to_run.json"
PROBES = ("abstract", "pretext", "proof", "instant", "quiz", "quiz_abstract")
MODES = ("guided", "unguided")
QUIZ_OPTIONS = 5  # original + 3 perturbations + "none of the above"

MIN_TARGET_CHARS = 120
ABSTRACT_PREFIX_FRACTION = 0.4

# Wording matters: "verbatim"/"exact text"/"training data" trips recitation guardrails
# (Gemini refused ~35% of continuations with such wording, including famous papers).
PROMPT_VERSION = 2
RECALL_SYSTEM = (
    "You are helping to reconstruct a passage from a document. You are given the beginning of the "
    "passage; write the text that follows it in the original. If you remember the document, "
    "reproduce the continuation as closely as you can; otherwise write the most likely continuation. "
    "Output only the continuation text, with no commentary and no restatement of the given beginning."
)
INSTANT_SYSTEM = (
    "You are taking a recall test. Answer the question immediately with the final answer only. "
    "Do not derive, compute, or explain anything. If you do not know the answer, say NONE."
)
QUIZ_ABSTRACT_SYSTEM = (
    "You are taking a recognition test about arXiv papers. Exactly one of the options may be the "
    "verbatim abstract of the named paper; the others are perturbed. Reply with the letter of the "
    "option that is the paper's actual abstract, or E if you believe none of them is. "
    "Reply with a single letter."
)
QUIZ_SYSTEM = (
    "You are taking a recognition test about the RealMath benchmark (a public dataset of "
    "questions extracted from arXiv papers). Exactly one of the options may be the verbatim "
    "wording of a RealMath question; the others are perturbed. Reply with the letter of the "
    "option that appears verbatim in the benchmark, or E if you believe none of them does. "
    "Reply with a single letter."
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$(])")


@dataclass
class ProbeRecord:
    probe_id: str
    probe: str
    mode: str
    group: str
    arxiv_id: str
    cohort: str | None
    problem_id: str | None
    system: str
    user: str
    target: str | None  # text to compare against; the answer letter for quiz; None for instant
    max_tokens: int
    meta: dict[str, Any]
    prompt_version: int = PROMPT_VERSION


# --------------------------------------------------------------------------- text splitting


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]


def split_prefix(text: str, fraction: float) -> tuple[str, str] | None:
    """Split at a sentence boundary once the prefix holds ``fraction`` of the characters."""

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None
    cut = 0
    consumed = 0
    for index, sentence in enumerate(sentences[:-1]):
        consumed += len(sentence) + 1
        cut = index + 1
        if consumed >= fraction * len(text):
            break
    prefix = " ".join(sentences[:cut])
    target = " ".join(sentences[cut:])
    if len(target) < MIN_TARGET_CHARS or not prefix:
        return None
    return prefix, target


def continuation_budget(target: str) -> int:
    return max(64, int(len(target) / 3) + 64)


# --------------------------------------------------------------------------- quiz perturbations

SYNONYMS: tuple[tuple[str, str], ...] = (
    ("What is", "Determine"),
    ("Find", "Compute"),
    ("Compute", "Evaluate"),
    ("Determine", "Identify"),
    ("Let", "Suppose"),
    ("Consider", "Take"),
    ("Suppose", "Assume"),
    ("Assume", "Suppose"),
    ("denote", "write"),
    ("denotes", "stands for"),
    ("Provide", "Give"),
    ("Give", "State"),
    ("Express", "Write"),
    ("exactly", "precisely"),
    ("such that", "so that"),
    ("for all", "for every"),
    ("for every", "for all"),
    ("there exists", "there is"),
    ("in terms of", "as a function of"),
    ("the number of", "how many"),
    ("as a function of", "in terms of"),
    # abstract vocabulary
    ("We prove", "We show"),
    ("We show", "We prove"),
    ("we prove", "we show"),
    ("we show", "we prove"),
    ("In this paper", "In this article"),
    ("In this article", "In this paper"),
    ("we study", "we investigate"),
    ("we obtain", "we derive"),
    ("we establish", "we prove"),
    ("Moreover", "Furthermore"),
    ("Furthermore", "Moreover"),
    ("In particular", "Specifically"),
    ("Finally", "Lastly"),
    ("As an application", "As a consequence"),
    ("we introduce", "we define"),
    ("we consider", "we study"),
)


def _lexical_variant(text: str, rng: random.Random, *, changes: int) -> str:
    candidates = [(src, dst) for src, dst in SYNONYMS if re.search(r"\b" + re.escape(src) + r"\b", text)]
    rng.shuffle(candidates)
    result = text
    applied = 0
    for src, dst in candidates:
        if applied >= changes:
            break
        new = re.sub(r"\b" + re.escape(src) + r"\b", dst, result, count=1)
        if new != result:
            result = new
            applied += 1
    return result


def _reorder_variant(text: str) -> str:
    """Swap two adjacent sentences in the middle; moving the opening sentence reads unnatural."""

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return text
    i = max(0, len(sentences) // 2 - 1)
    sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]
    return " ".join(sentences)


def quiz_variants(question: str, seed: int) -> list[str] | None:
    """Three distinct perturbations of the question, or None if the text resists perturbation."""

    rng = random.Random(seed)
    candidates = [
        _lexical_variant(question, rng, changes=1),
        _lexical_variant(question, rng, changes=2),
        _reorder_variant(question),
        _lexical_variant(_reorder_variant(question), rng, changes=1),
        _lexical_variant(question, rng, changes=3),
    ]
    variants: list[str] = []
    for candidate in candidates:
        if candidate != question and candidate not in variants:
            variants.append(candidate)
        if len(variants) == 3:
            return variants
    return None


# --------------------------------------------------------------------------- record builders


def _guidance(paper: dict[str, Any]) -> str:
    return f"Source: arXiv:{paper['arxiv_id']}, \"{paper['title']}\" ({paper.get('published', '')[:4]}).\n\n"


def abstract_records(paper: dict[str, Any]) -> list[ProbeRecord]:
    if not paper.get("abstract"):
        return []
    split = split_prefix(paper["abstract"], ABSTRACT_PREFIX_FRACTION)
    split_method = "sentence"
    if split is None:
        # Retain short and single-sentence abstracts instead of silently dropping
        # papers from the sampled groups. Existing eligible prompts stay unchanged.
        abstract = paper["abstract"].strip()
        words = list(re.finditer(r"\S+", abstract))
        if len(words) < 2:
            raise ValueError(f"Cannot split abstract for {paper['arxiv_id']}: fewer than two words")
        desired = ABSTRACT_PREFIX_FRACTION * len(abstract)
        cut = min((word.start() for word in words[1:]), key=lambda pos: abs(pos - desired))
        split = abstract[:cut].rstrip(), abstract[cut:]
        split_method = "word_fallback"
    prefix, target = split
    records: list[ProbeRecord] = []
    for mode in MODES:
        head = _guidance(paper) if mode == "guided" else "The passage is the abstract of a scientific paper.\n\n"
        user = head + "Beginning of the abstract:\n\n" + prefix + "\n\nContinuation:"
        records.append(
            ProbeRecord(
                probe_id=f"abstract:{mode}:{paper['arxiv_id']}",
                probe="abstract",
                mode=mode,
                group=paper["group"],
                arxiv_id=paper["arxiv_id"],
                cohort=paper.get("cohort"),
                problem_id=None,
                system=RECALL_SYSTEM,
                user=user,
                target=target,
                max_tokens=continuation_budget(target),
                meta={"prefix_chars": len(prefix), "target_chars": len(target), "split_method": split_method},
            )
        )
    return records


def pretext_records(paper: dict[str, Any]) -> list[ProbeRecord]:
    if not paper.get("pretext"):
        return []
    split = split_prefix(paper["pretext"], 0.5)
    if split is None:
        return []
    prefix, target = split
    records: list[ProbeRecord] = []
    for mode in MODES:
        head = _guidance(paper) if mode == "guided" else "The passage is from a mathematics paper.\n\n"
        user = head + "Beginning of the passage (LaTeX source):\n\n" + prefix + "\n\nContinuation:"
        records.append(
            ProbeRecord(
                probe_id=f"pretext:{mode}:{paper['arxiv_id']}",
                probe="pretext",
                mode=mode,
                group=paper["group"],
                arxiv_id=paper["arxiv_id"],
                cohort=paper.get("cohort"),
                problem_id=None,
                system=RECALL_SYSTEM,
                user=user,
                target=target,
                max_tokens=continuation_budget(target),
                meta={"prefix_chars": len(prefix), "target_chars": len(target), "theorem_env": paper.get("theorem_env")},
            )
        )
    return records


def proof_records(paper: dict[str, Any]) -> list[ProbeRecord]:
    statement = paper.get("theorem_statement")
    target = paper.get("proof_start")
    if not statement or not target or len(target) < MIN_TARGET_CHARS:
        return []
    user = (
        _guidance(paper)
        + f"The paper states the following result (LaTeX source of the {paper.get('theorem_env') or 'theorem'} environment):\n\n"
        + statement
        + "\n\nWrite the opening of the proof as it appears in the paper, starting right after \\begin{proof}.\n\nProof opening:"
    )
    return [
        ProbeRecord(
            probe_id=f"proof:guided:{paper['arxiv_id']}",
            probe="proof",
            mode="guided",
            group=paper["group"],
            arxiv_id=paper["arxiv_id"],
            cohort=paper.get("cohort"),
            problem_id=None,
            system=RECALL_SYSTEM,
            user=user,
            target=target,
            max_tokens=continuation_budget(target),
            meta={"target_chars": len(target), "theorem_matched_by": paper.get("theorem_matched_by")},
        )
    ]


def instant_records(paper: dict[str, Any], problems: dict[str, dict[str, Any]]) -> list[ProbeRecord]:
    records: list[ProbeRecord] = []
    for problem_id in paper.get("problem_ids", []):
        problem = problems[problem_id]
        user = (
            "Question:\n\n"
            + problem["question"]
            + "\n\nReply with a JSON object {\"final_answer\": <the answer in LaTeX or plain text>, "
            "\"sympy_answer\": <the same answer as a SymPy-parsable string, or a list of strings for multi-part answers>}. "
            "No other text."
        )
        records.append(
            ProbeRecord(
                probe_id=f"instant:none:{problem_id}",
                probe="instant",
                mode="none",
                group=paper["group"],
                arxiv_id=paper["arxiv_id"],
                cohort=paper.get("cohort"),
                problem_id=problem_id,
                system=INSTANT_SYSTEM,
                user=user,
                target=None,
                max_tokens=200,
                meta={
                    "ground_truth": problem.get("original_answer"),
                    "ground_truth_sympy_answer": problem.get("sympy_answer"),
                    "answer_kind": problem.get("answer_kind"),
                },
            )
        )
    return records


def quiz_records(paper: dict[str, Any], problems: dict[str, dict[str, Any]], *, seed: int) -> list[ProbeRecord]:
    records: list[ProbeRecord] = []
    for problem_id in paper.get("problem_ids", []):
        question = problems[problem_id]["question"]
        item_seed = seed * 100003 + int(re.sub(r"\D", "", problem_id) or 0)
        variants = quiz_variants(question, item_seed)
        if variants is None:
            continue
        rng = random.Random(item_seed)
        options = [question] + variants
        rng.shuffle(options)
        answer = "ABCD"[options.index(question)]
        user = "Which option is the verbatim RealMath question?\n\n" + "\n\n".join(
            f"({letter}) {text}" for letter, text in zip("ABCD", options)
        ) + "\n\n(E) None of the above.\n\nAnswer:"
        records.append(
            ProbeRecord(
                probe_id=f"quiz:none:{problem_id}",
                probe="quiz",
                mode="none",
                group=paper["group"],
                arxiv_id=paper["arxiv_id"],
                cohort=paper.get("cohort"),
                problem_id=problem_id,
                system=QUIZ_SYSTEM,
                user=user,
                target=answer,
                max_tokens=8,
                meta={"options": options},
            )
        )
    return records


def quiz_abstract_records(paper: dict[str, Any], *, seed: int) -> list[ProbeRecord]:
    abstract = paper.get("abstract") or ""
    if len(abstract) < 300:
        return []
    item_seed = seed * 100003 + int(re.sub(r"\D", "", paper["arxiv_id"]) or 0)
    variants = quiz_variants(abstract, item_seed)
    if variants is None:
        return []
    rng = random.Random(item_seed)
    options = [abstract] + variants
    rng.shuffle(options)
    answer = "ABCD"[options.index(abstract)]
    user = (
        f"Paper: arXiv:{paper['arxiv_id']}, \"{paper['title']}\" ({paper.get('published', '')[:4]}).\n\n"
        "Which option is the verbatim abstract of this paper?\n\n"
        + "\n\n".join(f"({letter}) {text}" for letter, text in zip("ABCD", options))
        + "\n\n(E) None of the above.\n\nAnswer:"
    )
    return [
        ProbeRecord(
            probe_id=f"quiz_abstract:none:{paper['arxiv_id']}",
            probe="quiz_abstract",
            mode="none",
            group=paper["group"],
            arxiv_id=paper["arxiv_id"],
            cohort=paper.get("cohort"),
            problem_id=None,
            system=QUIZ_ABSTRACT_SYSTEM,
            user=user,
            target=answer,
            max_tokens=8,
            meta={"options": options},
        )
    ]


def build_probes(corpus: dict[str, Any], problems: dict[str, dict[str, Any]], *, probes: Sequence[str], seed: int) -> list[ProbeRecord]:
    records: list[ProbeRecord] = []
    for paper in corpus["papers"]:
        if "abstract" in probes:
            records.extend(abstract_records(paper))
        if "pretext" in probes:
            records.extend(pretext_records(paper))
        if "proof" in probes:
            records.extend(proof_records(paper))
        if "quiz_abstract" in probes:
            records.extend(quiz_abstract_records(paper, seed=seed))
        if paper["group"] == "benchmark":
            if "instant" in probes:
                records.extend(instant_records(paper, problems))
            if "quiz" in probes:
                records.extend(quiz_records(paper, problems, seed=seed))
    return records


def load_problems(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in json.loads(path.read_text(encoding="utf-8"))}


def load_probes(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(records: Sequence[ProbeRecord]) -> dict[str, Any]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        key = f"{record.probe}/{record.mode}"
        summary.setdefault(key, {})
        summary[key][record.group] = summary[key].get(record.group, 0) + 1
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_DIR / "corpus.json")
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "probes.jsonl")
    parser.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=0, help="Print this many example prompts per probe.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    records = build_probes(corpus, load_problems(args.problems), probes=args.probes, seed=args.seed)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    print(json.dumps(summarize(records), indent=2))
    approx_tokens = sum(len(r.system) + len(r.user) for r in records) / 4 + sum(r.max_tokens for r in records)
    print(f"{len(records)} probes -> {args.output}; ~{approx_tokens / 1000:.0f}k tokens per model (upper bound)")
    if args.show:
        shown: dict[str, int] = {}
        for record in records:
            if shown.get(record.probe, 0) >= args.show:
                continue
            shown[record.probe] = shown.get(record.probe, 0) + 1
            print("=" * 80, f"\n[{record.probe_id}] target={str(record.target)[:100]!r}\n--- system\n{record.system}\n--- user\n{record.user}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
