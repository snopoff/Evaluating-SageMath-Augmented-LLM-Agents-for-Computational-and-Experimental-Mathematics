import rootutils
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping


from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

ROOT = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.utils.console_logging import ConsoleLogger  # noqa: E402

load_dotenv()

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 16192

DEFAULT_INPUT_PATH = Path("data/processed/realmathbenchmark_full.json")
DEFAULT_OUTPUT_PATH = Path(f"data/processed/realmathbenchmark_full_preprocessed_{time.strftime('%Y-%m-%d-%H-%M')}.json")
DEFAULT_START_INDEX = 9
DEFAULT_LIMIT = 1

DOCUMENT_BEGIN_MARKER = r"\begin{document}"
BODY_START_MARKERS = (
    r"\section{",
    r"\section*{",
    r"\subsection{",
    r"\subsection*{",
    r"\begin{definition}",
    r"\begin{defn}",
    r"\begin{notation}",
    r"\begin{theorem}",
    r"\begin{thm}",
    r"\begin{lemma}",
    r"\begin{lem}",
    r"\begin{proposition}",
    r"\begin{prop}",
    r"\begin{problem}",
    r"\begin{question}",
)
BODY_END_MARKERS = (
    r"\begin{thebibliography}",
    r"\bibliography{",
    r"\bibliographystyle{",
    r"\end{document}",
)
DROP_WHOLE_ENVIRONMENTS = (
    "abstract",
    "figure",
    "figure*",
    "table",
    "table*",
    "tikzpicture",
    "thebibliography",
)
DROP_WRAPPER_ENVIRONMENTS = (
    "center",
    "flushleft",
    "flushright",
)
DROP_COMMAND_PATTERNS = (
    r"\\documentclass(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\usepackage(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\(?:bibliography|bibliographystyle|label|caption)(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\cite\w*(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\(?:url|href)(?:\[[^\]]*\])?\{[^{}]*\}(?:\{[^{}]*\})?",
    r"\\(?:maketitle|tableofcontents|listoffigures|listoftables|newpage|clearpage|pagebreak|medskip|smallskip|bigskip|noindent)\b",
    r"\\(?:vspace|hspace)\*?(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\(?:footnote|thanks)\{[^{}]*\}",
)
DROP_STYLE_COMMANDS = (
    "footnotesize",
    "small",
    "normalsize",
    "large",
    "Large",
    "LARGE",
    "huge",
    "Huge",
    "bf",
    "bfseries",
    "it",
    "itshape",
    "rm",
    "rmfamily",
    "tt",
    "ttfamily",
    "scshape",
    "slshape",
    "em",
    "qquad",
    "quad",
    "hfill",
    "vfill",
)


class ProblemPreprocessingResult(BaseModel):
    """Submit the classification and self-contained rewritten question."""

    model_config = ConfigDict(extra="forbid")

    revised_question: str = Field(
        description=(
            "A self-contained rewrite of the original question that includes the paper-specific "
            "definitions, notation, hypotheses, and referenced formulas needed to answer it."
        )
    )
    difficulty: Literal["easy", "middle", "hard"] = Field(description="Overall difficulty of the problem.")
    difficulty_explanation: str = Field(description="Very brief justification for the difficulty verdict (one short sentence).")
    confidence: int = Field(
        ge=1,
        le=5,
        description="Confidence in the overall classification on a 1 to 5 scale.",
    )
    answer_type: Literal["number", "expression", "formula"] = Field(
        description=(
            "Type of the reference answer. "
            "'number' = a concrete numeric value (e.g. 0, 24). "
            "'expression' = a symbolic expression in some variables (e.g. 2^k, (2s-1)r^2-s-1, m+n). "
            "'formula' = an equality of the form 'A = B' where A is a named characteristic / "
            "invariant and B is an expression (e.g. |RAI_{2n-1}(4321)| = M_n - M_{n-2})."
        )
    )
    answer_type_explanation: str = Field(description="Very brief justification for the answer_type verdict (one short sentence).")


PROMPT_TEMPLATE = """You preprocess a mathematics problem.

For the given problem (paper context, theorem statement, question, reference answer), do two things:

A. Rewrite the question into a field called revised_question.

The revised_question must be self-contained and computationally usable: a tool-augmented mathematical reasoning agent should be able to understand the problem and run symbolic or numerical experiments without access to the original paper.

Using the provided context, inline the paper-specific information required to interpret the question and reconstruct the mathematical objects that appear in it.

Requirements:
- Preserve the exact mathematical meaning and target of the original question.
- Preserve original notation and LaTeX whenever useful.
- Preserve explicit mathematical definitions and formulas that are directly relevant to the question.
- Resolve references to previously defined notation, equations, assumptions, constructions, operators, spaces, maps, matrices, modules, invariants, or named results when needed for understanding.
- Include enough detail for computational instantiation of the problem in a CAS such as SageMath.
- Prefer concise local definitions over large copied passages.
- Do not include unrelated background, motivation, proofs, examples, literature discussion, or auxiliary results not needed to interpret the question.
- Do not answer the question.
- Do not add hints, explanations, conjectures, or solution steps.
- If the original question is already self-contained, do not make unnecessary edits and return it as-is.

The revised_question field must contain only the rewritten question, not commentary about the rewrite.

B. Classify the problem along two independent axes:

1. difficulty: "easy" | "middle" | "hard"
   IMPORTANT: every problem in this dataset is a theorem / question taken from a REAL published
   mathematics article, so the scale is RELATIVE to the population of research-paper results — not
   absolute. By definition almost nothing here is "easy" in the school-olympiad sense. Calibrate as:
   - easy: among the simpler results in a research paper — a direct corollary, a base-case lemma,
     a routine computation by definition, or a result whose proof is a few lines of standard
     manipulation that a competent graduate student would reproduce on the spot.
   - middle: a typical lemma / proposition from a research paper — requires combining several
     standard techniques, careful case analysis, an inductive argument, or moderate algebraic /
     combinatorial bookkeeping; the kind of result that takes a working researcher a session of
     focused effort but uses no deep machinery beyond the field's standard toolkit.
   - hard: among the harder results in a research paper — main theorems requiring genuinely
     non-trivial insight, long technical derivations, deep machinery, specialized prior results,
     or several non-obvious ingredients combined in a non-routine way.
   Judge difficulty by the proof / derivation needed to GET the reference answer, not by how
   intimidating the LaTeX looks.

2. answer_type: "number" | "expression" | "formula"
   - number: a concrete numeric value (e.g. 0, 24, 1/2).
   - expression: a symbolic expression in some variables (e.g. 2^k, (2s-1)r^2-s-1, m+n).
   - formula: an equality "A = B" where A is a named characteristic / invariant / quantity and B is an expression
     (e.g. |RAI_{{2n-1}}(4321)| = M_n - M_{{n-2}}, H_G(u) = (2H_{{m-1}} + 1/m)/(2m-1)).
   Look at the actual reference answer to decide.

Also report:
- confidence: integer 1-5 (1 = pure guess, 5 = certain) — overall confidence in your classification.
- For each verdict, a VERY brief one-sentence justification.

You MUST return the final answer as structured output matching the required preprocessing schema.
Do not include any extra prose outside that structured output.

Problem id: {problem_id}

Paper context (LaTeX):
{context}

Theorem statement (LaTeX):
{theorem}

Question:
{question}

Reference answer:
{answer}
"""


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


@lru_cache(maxsize=None)
def _classifier(model: str, max_tokens: int) -> Any:
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    llm = ChatAnthropic(
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
    )
    return llm.with_structured_output(ProblemPreprocessingResult, include_raw=True, method="function_calling")


def _usage_from_message(message: AIMessage | None) -> Usage:
    usage_candidates: list[Mapping[str, Any]] = []
    if message is not None:
        if isinstance(message.usage_metadata, Mapping):
            usage_candidates.append(message.usage_metadata)

        response_metadata = getattr(message, "response_metadata", {})
        if isinstance(response_metadata, Mapping):
            token_usage = response_metadata.get("token_usage")
            if isinstance(token_usage, Mapping):
                usage_candidates.append(token_usage)
            usage = response_metadata.get("usage")
            if isinstance(usage, Mapping):
                usage_candidates.append(usage)

        additional_kwargs = getattr(message, "additional_kwargs", {})
        if isinstance(additional_kwargs, Mapping):
            usage = additional_kwargs.get("usage")
            if isinstance(usage, Mapping):
                usage_candidates.append(usage)

    def first_int(*keys: str) -> int:
        for usage in usage_candidates:
            for key in keys:
                value = usage.get(key)
                if isinstance(value, int):
                    return value
        return 0

    return Usage(
        prompt_tokens=first_int("input_tokens", "prompt_tokens"),
        completion_tokens=first_int("output_tokens", "completion_tokens"),
    )


def _raw_message_preview(message: AIMessage | None) -> str:
    if message is None:
        return ""
    payload = {
        "content": message.content,
        "tool_calls": getattr(message, "tool_calls", None),
        "response_metadata": getattr(message, "response_metadata", {}),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.type,
        "content": _message_text(message),
    }
    if isinstance(message, AIMessage):
        payload["tool_calls"] = [
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "args": call.get("args", {}),
            }
            for call in (message.tool_calls or [])
        ]
    return payload


def _usage_payload(usage: Usage) -> dict[str, int]:
    total_tokens = usage.prompt_tokens + usage.completion_tokens
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": total_tokens,
    }


def _remove_environment(text: str, environment: str) -> str:
    pattern = rf"\\begin\{{{re.escape(environment)}\}}.*?\\end\{{{re.escape(environment)}\}}"
    return re.sub(pattern, " ", text, flags=re.DOTALL)


def clean_latex_context(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    if DOCUMENT_BEGIN_MARKER in cleaned:
        cleaned = cleaned.split(DOCUMENT_BEGIN_MARKER, 1)[1]

    body_start_positions = [cleaned.find(marker) for marker in BODY_START_MARKERS if cleaned.find(marker) >= 0]
    if body_start_positions:
        cleaned = cleaned[min(body_start_positions) :]

    body_end_positions = [cleaned.find(marker) for marker in BODY_END_MARKERS if cleaned.find(marker) >= 0]
    if body_end_positions:
        cleaned = cleaned[: min(body_end_positions)]

    for environment in DROP_WHOLE_ENVIRONMENTS:
        cleaned = _remove_environment(cleaned, environment)

    for environment in DROP_WRAPPER_ENVIRONMENTS:
        cleaned = cleaned.replace(rf"\begin{{{environment}}}", " ")
        cleaned = cleaned.replace(rf"\end{{{environment}}}", " ")

    for pattern in DROP_COMMAND_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.DOTALL)

    for command in DROP_STYLE_COMMANDS:
        cleaned = re.sub(rf"\\{re.escape(command)}\b", " ", cleaned)

    cleaned = cleaned.replace(r"\\", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _call_anthropic_once(
    prompt: str,
    model: str,
    max_tokens: int,
    *,
    logger: ConsoleLogger | None = None,
    agent_id: str = "preprocess_data",
    turn: int = 1,
) -> tuple[str, Usage]:
    """Single forced structured-output call through LangChain + Anthropic."""
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]
    last_raw: AIMessage | None = None
    last_error: BaseException | None = None
    last_usage = Usage(0, 0)

    for _ in range(3):
        response = _classifier(model, max_tokens).invoke(messages)
        raw = response.get("raw")
        raw_message = raw if isinstance(raw, AIMessage) else None
        parsed = response.get("parsed")
        parsing_error = response.get("parsing_error")

        last_raw = raw_message
        last_error = parsing_error
        last_usage = _usage_from_message(raw_message)

        if parsing_error is None and parsed is not None:
            try:
                result = parsed if isinstance(parsed, ProblemPreprocessingResult) else ProblemPreprocessingResult.model_validate(parsed)
            except Exception as exc:  # noqa: BLE001 - preserve the model's raw response below
                last_error = exc
            else:
                if logger is not None:
                    logger.log_model_call(
                        agent_id=agent_id,
                        turn=turn,
                        model_name=model,
                        messages=[_message_payload(message) for message in messages],
                        raw_response=_raw_message_preview(raw_message),
                        parsed_payload=result.model_dump(),
                        token_usage=_usage_payload(last_usage),
                    )
                return json.dumps(result.model_dump(), ensure_ascii=False), last_usage

        messages.append(
            HumanMessage(
                content=(
                    "Return a valid structured preprocessing result matching the schema exactly. "
                    "Do not include plain text outside the structured output."
                )
            )
        )

    error_payload = {
        "_parse_error": str(last_error or "structured output parsing failed"),
        "_raw": _raw_message_preview(last_raw),
    }
    if logger is not None:
        logger.log_model_call(
            agent_id=agent_id,
            turn=turn,
            model_name=model,
            messages=[_message_payload(message) for message in messages],
            raw_response=_raw_message_preview(last_raw),
            parsed_payload=error_payload,
            token_usage=_usage_payload(last_usage),
        )
    return json.dumps(error_payload, ensure_ascii=False), last_usage


def load_problems(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() != ".jsonl" and text[0] in "[{":
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("results", "data", "problems"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return rows
        raise ValueError(f"Unsupported JSON dataset shape in {path}")

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected each JSONL row in {path} to be an object")
        rows.append(payload)
    return rows


def build_prompt(problem: dict[str, Any], *, context: str | None = None) -> str:
    prompt_context = context
    if prompt_context is None:
        raw_context = str(problem.get("context", "")).strip()
        prompt_context = clean_latex_context(raw_context) or raw_context

    return PROMPT_TEMPLATE.format(
        problem_id=problem.get("id", ""),
        context=prompt_context,
        theorem=str(problem.get("theorem", "")).strip(),
        question=str(problem.get("question", "")).strip(),
        answer=str(problem.get("answer", "")).strip(),
    )


def parse_preprocessing_result(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"_parse_error": f"JSONDecodeError: {exc}", "_raw": raw}
    if not isinstance(payload, dict):
        return {"_parse_error": "top-level not an object", "_raw": raw}
    return payload


def preprocess_problem(
    problem: dict[str, Any],
    *,
    logger: ConsoleLogger | None = None,
    agent_id: str = "preprocess_data",
    turn: int = 1,
) -> dict[str, Any]:
    raw_context = str(problem.get("context", "")).strip()
    cleaned_context = clean_latex_context(raw_context) or raw_context
    prompt = build_prompt(problem, context=cleaned_context)
    started = time.perf_counter()
    raw, usage = _call_anthropic_once(
        prompt=prompt,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        logger=logger,
        agent_id=agent_id,
        turn=turn,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    parsed = parse_preprocessing_result(raw or "")
    llm_payload = {
        "model": MODEL,
        "revised_question": parsed.get("revised_question"),
        "difficulty": parsed.get("difficulty"),
        "difficulty_explanation": parsed.get("difficulty_explanation"),
        "confidence": parsed.get("confidence"),
        "answer_type": parsed.get("answer_type"),
        "answer_type_explanation": parsed.get("answer_type_explanation"),
        "raw_response": raw,
        "parse_error": parsed.get("_parse_error"),
        "context_chars_raw": len(raw_context),
        "context_chars_cleaned": len(cleaned_context),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "elapsed_ms": elapsed_ms,
    }

    if logger is not None:
        logger.log_solve_result(
            agent_id=agent_id,
            final_answer=str(llm_payload.get("revised_question") or ""),
            turn_count=1,
            stop_reason="preprocessed" if not llm_payload.get("parse_error") else "parse_error",
            tool_traces=[],
            token_usage=_usage_payload(usage),
            explanation=str(llm_payload.get("difficulty_explanation") or ""),
            confidence=llm_payload.get("confidence") if isinstance(llm_payload.get("confidence"), int) else 0,
            final_payload=llm_payload,
        )

    return {
        "id": problem.get("id"),
        "question": problem.get("question"),
        "theorem": problem.get("theorem"),
        "answer": problem.get("answer"),
        "llm": llm_payload,
    }


CONSOLE_FIELD_LIMIT = 250


def _truncate(value: Any, limit: int = CONSOLE_FIELD_LIMIT) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def save_results(path: Path, results: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess dataset problems by rewriting questions and classifying difficulty / answer-type via LangChain + Anthropic."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to the input dataset JSON / JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write the preprocessed output JSON.")
    parser.add_argument(
        "--logger-output-dir",
        type=Path,
        default=None,
        help="Optional directory for ConsoleLogger console/trace artifacts.",
    )
    args = parser.parse_args()

    ConsoleLogger.setup_logging()
    logger = ConsoleLogger(mode="preprocess_data")

    problems = load_problems(args.input)
    problems = problems[DEFAULT_START_INDEX:] if DEFAULT_START_INDEX and DEFAULT_START_INDEX > 0 else problems
    problems = problems[:DEFAULT_LIMIT] if DEFAULT_LIMIT and DEFAULT_LIMIT > 0 else problems

    logger.start_run(
        metadata={
            "agent_id": "preprocess_data",
            "model": MODEL,
            "input_path": str(args.input),
            "output_path": str(args.output),
            "limit": DEFAULT_LIMIT,
            "total": len(problems),
        }
    )
    logger.progress(f"Loaded {len(problems)} problems from {args.input}")

    results: list[dict[str, Any]] = []
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "easy": 0,
        "middle": 0,
        "hard": 0,
        "number": 0,
        "expression": 0,
        "formula": 0,
        "errors": 0,
    }

    separator = "_" * 80

    for index, problem in enumerate(problems, start=1):
        problem_id = problem.get("id", f"row-{index:05d}")
        theorem = str(problem.get("theorem", "")).strip()
        context = str(problem.get("context", "")).strip()
        cleaned_context = clean_latex_context(context) or context
        question = str(problem.get("question", "")).strip()
        reference_answer = str(problem.get("answer", "")).strip()

        agent_id = f"preprocess_data:{problem_id}"

        logger.log(separator, level="separator", color="white")
        logger.progress(f"Problem [{index}/{len(problems)}] id={problem_id}")
        logger.log(
            f"\nContext:\n{_truncate(cleaned_context)}",
            level="input",
            color="white",
        )
        logger.log(
            f"  context chars raw={len(context)} cleaned={len(cleaned_context)}",
            level="input",
            color="white",
        )
        logger.log(f"\nTheorem:\n{_truncate(theorem)}", level="input", color="white")
        logger.log(f"\nQuestion:\n{_truncate(question)}", level="input", color="white")
        logger.log(f"\nReference answer:\n{_truncate(reference_answer)}", level="input", color="white")

        try:
            record = preprocess_problem(problem, logger=logger, agent_id=agent_id, turn=index)
        except Exception as exc:  # noqa: BLE001 — keep going on a single failure
            logger.log(f"\nLLM answer:\n  ERROR: {_truncate(exc)}", level="error", color="red")
            record = {
                "id": problem_id,
                "question": problem.get("question"),
                "theorem": problem.get("theorem"),
                "answer": problem.get("answer"),
                "llm": {"model": MODEL, "error": str(exc)},
            }
            logger.log_solve_result(
                agent_id=agent_id,
                final_answer="",
                turn_count=0,
                stop_reason="error",
                tool_traces=[],
                token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                explanation=str(exc),
                confidence=0,
                final_payload=record["llm"],
            )
            totals["errors"] += 1
        else:
            llm = record["llm"]
            totals["prompt_tokens"] += llm.get("prompt_tokens") or 0
            totals["completion_tokens"] += llm.get("completion_tokens") or 0
            difficulty = llm.get("difficulty")
            if difficulty in totals:
                totals[difficulty] += 1
            answer_type = llm.get("answer_type")
            if answer_type in totals:
                totals[answer_type] += 1

            logger.log("\nLLM answer:", level="output", color="green")
            logger.log(f"  revised_question: {_truncate(llm.get('revised_question'))}", level="output", color="green")
            logger.log(f"  difficulty: {_truncate(difficulty)}", level="output", color="green")
            logger.log(f"    why: {_truncate(llm.get('difficulty_explanation'))}", level="output", color="green")
            logger.log(f"  confidence: {llm.get('confidence')}", level="output", color="green")
            logger.log(f"  answer_type: {_truncate(answer_type)}", level="output", color="green")
            logger.log(f"    why: {_truncate(llm.get('answer_type_explanation'))}", level="output", color="green")
            logger.log(
                f"  (input_tokens={llm.get('prompt_tokens')}, output_tokens={llm.get('completion_tokens')}, {llm.get('elapsed_ms')} ms)",
                level="output",
                color="green",
            )

        results.append(record)
        save_results(
            args.output,
            results,
            meta={
                "model": MODEL,
                "input_path": str(args.input),
                "completed": len(results),
                "total": len(problems),
                "totals": totals,
            },
        )

    status = "completed" if totals["errors"] == 0 else "completed_with_errors"
    logger.finish_run(status=status)
    logger.progress(
        f"Done. easy={totals['easy']} middle={totals['middle']} hard={totals['hard']}  "
        f"number={totals['number']} expression={totals['expression']} formula={totals['formula']}  "
        f"errors={totals['errors']}  "
        f"prompt_tokens={totals['prompt_tokens']} completion_tokens={totals['completion_tokens']}"
    )
    logger.progress(f"Wrote {args.output}")
    if args.logger_output_dir is not None:
        saved_files = logger.save_output(args.logger_output_dir, prefix=args.output.stem, status=status)
        logger.progress(f"Wrote ConsoleLogger trace {saved_files['trace_json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
