"""Query models with the contamination probes and store raw responses.

Models are given as ``provider=model_name`` specs resolved through ``configs/model``
(same syntax as the judge scripts).  Reasoning is switched off where the provider
allows it, because the probes measure recall, not problem solving; check the
``reasoning_control`` line printed at start-up and the ``response_tokens`` of the
pilot to confirm the model did not think silently.

Responses are appended to ``responses/<label>.jsonl`` and the script resumes by
``probe_id``, so an interrupted run can simply be restarted.

Usage::

    # Pilot: a few papers per group, one model, every probe, print prompts first.
    uv run python src/benchmark/contamination/run.py --models openai=gpt-5.5 --pilot 5 --dry-run
    uv run python src/benchmark/contamination/run.py --models openai=gpt-5.5 --pilot 5

    # Full run, four models in parallel; GPT gets a 65-item random quiz subset (budget).
    uv run python src/benchmark/contamination/run.py --models openai=gpt-5.5 anthropic=claude-opus-4-8 \\
        openrouter=deepseek/deepseek-v4-pro google=gemini-3.1-pro-preview --workers 4 \\
        --quiz-limit 65 --quiz-limit-models openai=gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import rootutils
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
load_dotenv()

from src.benchmark.contamination.probes import PROBES, load_probes  # noqa: E402

QUIZ_PROBES = ("quiz", "quiz_abstract")
from src.benchmark.judge_sympy_answers import (  # noqa: E402
    _usage_from_message,
    instantiate_model_from_config,
    parse_judge_spec,
)

DEFAULT_DIR = ROOT / "data/reviews/contamination"
DEFAULT_MODELS = (
    "openai=gpt-5.5",
    "anthropic=claude-opus-4-8",
    "openrouter=deepseek/deepseek-v4-pro",
    "google=gemini-3.1-pro-preview",
)


def label_for(spec_value: str) -> str:
    spec = parse_judge_spec(spec_value)
    return f"{spec.provider}__{spec.model_name}".replace("/", "_")


def disable_reasoning(model: BaseChatModel) -> str:
    """Best-effort: turn off extended thinking so the answer reflects recall only."""

    module = type(model).__module__
    notes: list[str] = []
    try:
        if module.startswith("langchain_anthropic"):
            # Opus 4.8 rejects `temperature` ("deprecated for this model"), so only thinking is set.
            model.thinking = {"type": "disabled"}
            notes.append("anthropic thinking=disabled")
        elif module.startswith("langchain_openai") and "openrouter" in str(getattr(model, "openai_api_base", "") or ""):
            # OpenRouter's unified reasoning switch works across DeepSeek/Qwen/etc.
            model.extra_body = {**(model.extra_body or {}), "reasoning": {"enabled": False}}
            model.temperature = 0
            notes.append("openrouter reasoning.enabled=false, temperature=0")
        elif module.startswith("langchain_openai"):
            # Reasoning models reject `temperature`, so only the effort is set;
            # gpt-5.5 accepts none/low/medium/high/xhigh ('minimal' is rejected).
            model.reasoning_effort = "none"
            notes.append("openai reasoning_effort=none")
        elif module.startswith("langchain_qwq"):
            model.enable_thinking = False
            model.temperature = 0
            notes.append("qwen enable_thinking=False, temperature=0")
        elif module.startswith("langchain_google_genai"):
            # Gemini 3.1 Pro rejects "minimal"; "low" is the lowest level it accepts.
            model.thinking_level = "low"
            model.temperature = 0
            notes.append("google thinking_level=low, temperature=0 (lowest level this model accepts)")
        elif module.startswith("langchain_deepseek"):
            model.temperature = 0
            notes.append("deepseek temperature=0 (use deepseek-chat, not the reasoner)")
        else:
            notes.append(f"no reasoning control for {module}")
    except Exception as exc:  # noqa: BLE001 - report and continue with provider defaults
        notes.append(f"could not set reasoning controls: {exc}")
    return "; ".join(notes)


def message_text(message: Any) -> str:
    """Plain text of a reply; Gemini returns a list of content blocks instead of a string."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)


def load_done(path: Path, prompt_version: int) -> set[str]:
    """Probe ids answered under this prompt version; failures and stale prompts are re-run."""

    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("error") and row.get("prompt_version", 1) == prompt_version:
                done.add(row["probe_id"])
    return done


def select_pilot(records: list[dict[str, Any]], per_group: int) -> list[dict[str, Any]]:
    """Keep every probe for the first ``per_group`` papers of each group."""

    papers_by_group: dict[str, list[str]] = {}
    for record in records:
        papers = papers_by_group.setdefault(record["group"], [])
        if record["arxiv_id"] not in papers:
            papers.append(record["arxiv_id"])
    keep = {(group, arxiv_id) for group, papers in papers_by_group.items() for arxiv_id in papers[:per_group]}
    return [record for record in records if (record["group"], record["arxiv_id"]) in keep]


def query(model: BaseChatModel, record: dict[str, Any]) -> dict[str, Any]:
    messages = [SystemMessage(content=record["system"]), HumanMessage(content=record["user"])]
    started = time.monotonic()
    try:
        response = model.invoke(messages)
        usage = _usage_from_message(response)
        return {
            "probe_id": record["probe_id"],
            "response": message_text(response),
            "prompt_tokens": usage.prompt_tokens,
            "response_tokens": usage.completion_tokens,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - keep the failure in the log, continue the run
        return {
            "probe_id": record["probe_id"],
            "response": "",
            "prompt_tokens": 0,
            "response_tokens": 0,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def run_model(spec_value: str, records: list[dict[str, Any]], *, output_dir: Path, workers: int, max_tokens_cap: int) -> None:
    spec = parse_judge_spec(spec_value)
    label = label_for(spec_value)
    output_path = output_dir / f"{label}.jsonl"
    prompt_version = max(record.get("prompt_version", 1) for record in records)
    done = load_done(output_path, prompt_version)
    todo = [record for record in records if record["probe_id"] not in done]
    print(f"[{label}] {len(todo)} probes to run ({len(done)} already done)")
    if not todo:
        return

    model = instantiate_model_from_config(spec.provider, spec.model_name, max_tokens=max_tokens_cap)
    print(f"[{label}] reasoning_control: {disable_reasoning(model)}")

    lock = threading.Lock()
    failures = 0
    with output_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(query, model, record): record for record in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            result["model"] = label
            result["prompt_version"] = futures[future].get("prompt_version", 1)
            with lock:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            if result["error"]:
                failures += 1
                print(f"[{label}] {index}/{len(todo)} {result['probe_id']} ERROR {result['error'][:120]}")
            elif index % 25 == 0 or index == len(todo):
                print(f"[{label}] {index}/{len(todo)} done, last={result['probe_id']} ({result['response_tokens']} tok)")
    print(f"[{label}] finished, {failures} failures -> {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes-file", type=Path, default=DEFAULT_DIR / "probes.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR / "responses")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="provider=model_name specs.")
    parser.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES))
    parser.add_argument("--groups", nargs="+", default=None, help="Restrict to these corpus groups.")
    parser.add_argument("--pilot", type=int, default=0, help="Only the first N papers of every group.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many probes (0 = all).")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle probes before --limit (random subset instead of the first papers).")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent API calls per model (models themselves run in parallel).")
    parser.add_argument("--quiz-limit", type=int, default=0, help="Run only this many (shuffled) probes of each quiz type; 0 = all.")
    parser.add_argument("--quiz-limit-models", nargs="*", default=None, help="Apply --quiz-limit only to these model specs (default: all).")
    parser.add_argument("--max-tokens-cap", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Print the selection and the first prompt, no API calls.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = [r for r in load_probes(args.probes_file) if r["probe"] in args.probes]
    if args.groups:
        records = [r for r in records if r["group"] in args.groups]
    if args.pilot:
        records = select_pilot(records, args.pilot)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(records)
    if args.limit:
        records = records[: args.limit]

    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['probe']}/{record['mode']}/{record['group']}"
        counts[key] = counts.get(key, 0) + 1
    approx = sum(len(r["system"]) + len(r["user"]) for r in records) / 4 + sum(min(r["max_tokens"], args.max_tokens_cap) for r in records)
    print(json.dumps(counts, indent=2))
    print(f"{len(records)} probes per model, ~{approx / 1000:.0f}k tokens per model (upper bound), models={args.models}")
    if args.dry_run:
        if records:
            print("--- first prompt ---\n" + records[0]["system"] + "\n\n" + records[0]["user"])
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quiz_limited = set(args.quiz_limit_models or args.models) if args.quiz_limit else set()

    def records_for(spec_value: str) -> list[dict[str, Any]]:
        if spec_value not in quiz_limited:
            return records
        keep: set[str] = set()
        for probe in QUIZ_PROBES:
            quiz = [r for r in records if r["probe"] == probe]
            random.Random(args.shuffle_seed or 0).shuffle(quiz)
            keep |= {r["probe_id"] for r in quiz[: args.quiz_limit]}
        return [r for r in records if r["probe"] not in QUIZ_PROBES or r["probe_id"] in keep]

    # Models run concurrently (each with its own worker pool); the per-model log lines are prefixed.
    with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
        futures = {
            pool.submit(
                run_model,
                spec_value,
                records_for(spec_value),
                output_dir=args.output_dir,
                workers=args.workers,
                max_tokens_cap=args.max_tokens_cap,
            ): spec_value
            for spec_value in args.models
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - one model failing must not hide the others
                print(f"[{label_for(futures[future])}] FAILED: {exc.__class__.__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
