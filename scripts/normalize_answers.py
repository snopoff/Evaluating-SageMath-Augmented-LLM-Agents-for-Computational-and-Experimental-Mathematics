from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import rootutils
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sympy.parsing.sympy_parser import parse_expr

ROOT = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.utils.console_logging import ConsoleLogger  # noqa: E402

load_dotenv()

API_KEY = os.environ.get("YANDEX_API_KEY", "")
MODEL = f"gpt://{os.environ.get('YANDEX_FOLDER_ID')}/deepseek-v32/latest"
BASE_URL = "https://ai.api.cloud.yandex.net/v1"
MAX_COMPLETION_TOKENS = 4096
MAX_RETRIES = 3
TEMPERATURE = 0.0

DEFAULT_INPUT_PATH = Path("data/processed/medium_hard_number_expression.json")
DEFAULT_OUTPUT_DIR = Path("outputs/normalize_answers")
DEFAULT_LIMIT = -1
DEFAULT_SLEEP_SEC = 0.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SEC = 10.0
FILE_TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M"

PROMPT_TEMPLATE = """You normalize mathematical reference answers into canonical SymPy strings for downstream evaluation.

Return exactly one JSON object with exactly these two keys:
- "answer_kind": either "single" or "multi"
- "sympy_answer":
  - if "answer_kind" is "single": one SymPy-parsable string
  - if "answer_kind" is "multi": a non-empty JSON list of SymPy-parsable strings

Hard requirements:
- Return JSON only.
- Do not return markdown fences, prose, commentary, labels such as "Answer:", or any extra keys.
- Never return an empty response.
- If the answer can be represented faithfully in SymPy, you must return a valid JSON object.

Core rules:
- Output valid SymPy syntax, never LaTeX.
- Extract the mathematical answer from surrounding prose.
- Preserve the mathematical content exactly. Do not drop conditions, dependencies, bounds, indices, or solution structure.
- If the source is of the form lhs = rhs and lhs is only a label/name for the answer, return rhs only.
- Use Eq(lhs, rhs) only when the equality itself is mathematically part of the answer, or when preserving several named quantities is necessary.
- If the answer is one mathematical object, return "answer_kind": "single".
- If the answer contains multiple distinct mathematical objects that should remain separate and ordered, return "answer_kind": "multi".

Safe identifier rules:
- Never use SymPy built-ins, reserved names, or Python keywords as free identifiers.
- Rewrite unsafe free identifiers deterministically with the suffix "_symbol".
- Examples:
  beta -> beta_symbol
  gamma -> gamma_symbol
  lambda -> lambda_symbol
  zeta -> zeta_symbol
- Keep standard SymPy constants and built-ins unchanged, including:
  pi, E, I, oo, True, False
- Use ASCII identifiers only.

SymPy syntax rules:
- Use "**" for exponentiation, never "^".
- Use explicit multiplication, such as 2*m, never 2m.
- Prefer exact forms: Rational(1, 2), pi, E, I, oo.
- Do not use decimal floats unless the original answer is explicitly approximate.
- Use appropriate SymPy constructors/functions when needed, including:
  Eq, Ne, Lt, Le, Gt, Ge, And, Or, Not,
  Piecewise, Tuple, FiniteSet, Union, Intersection, ImageSet, Lambda, Range, ConditionSet,
  Interval, Matrix, Sum, Product, Integral, Derivative, Limit,
  Mod, floor, ceiling, sqrt, Abs, Min, Max, binomial, factorial, fibonacci, asin, acos, atan, exp, log, LambertW

Conditions and relations:
- In symbolic conditions, use only SymPy boolean/relational syntax:
  And(...), Or(...), Not(...), Eq(...), Ne(...), Lt(...), Le(...), Gt(...), Ge(...), Mod(...)
- Do not use Python boolean operators `and`, `or`, `not` inside symbolic conditions.
- For inequalities, prefer Ge/Le/Gt/Lt form when normalization is clearer.
- For parity or divisibility conditions, use Mod(...), for example Eq(Mod(m, 2), 0).

Sets, tuples, and solution families:
- If the answer is a finite unordered solution set, prefer one single SymPy object:
  FiniteSet(...)
  and return "answer_kind": "single".
- If one item is naturally a point/vector/tuple, encode it as Tuple(...).
- If the answer uses sign shorthand such as (±a, ±b, ±c), expand all independent sign combinations explicitly unless the text states the signs are linked.
- If the answer is a union of parametrized families, use set constructors such as:
  Union(...), ImageSet(Lambda(...), Range(...)), FiniteSet(...)
- For integer parameters with lower bounds such as k >= 2, use Range(2, oo) when appropriate.
- Use "multi" only when the answer truly consists of several separate mathematical objects that should remain ordered, not when one FiniteSet(...) or Union(...) object captures the answer faithfully.

Named constants and implicit definitions:
- If an auxiliary constant is defined implicitly and a standard exact SymPy expression exists, prefer the explicit expression.
- Example:
  if gamma satisfies gamma*exp(1 + gamma) = 1, then use LambertW(exp(-1)) instead of a bare free identifier gamma.
- If no explicit standard expression is available, keep a safe symbol name such as gamma_symbol.

Examples:

Original: $$\pi(\beta-\alpha)^2$$
Output: {"answer_kind":"single","sympy_answer":"pi * (beta_symbol - alpha)**2"}

Original: $$\frac{1}{2\gamma}$$ where gamma satisfies gamma*exp(1 + gamma)=1
Output: {"answer_kind":"single","sympy_answer":"1/(2*LambertW(exp(-1)))"}

Original: \(\theta_1=\theta_2=\arcsin\gamma,\ \beta_1=\beta_2=\pi-\arcsin\gamma,\ \text{Optimal Objective Value}=\gamma^2\)
Output: {"answer_kind":"single","sympy_answer":"Tuple(asin(gamma_symbol), asin(gamma_symbol), pi - asin(gamma_symbol), pi - asin(gamma_symbol), gamma_symbol**2)"}

Original: \(\lambda^2 / (\kappa^2 \tau + \lambda)^2\)
Output: {"answer_kind":"single","sympy_answer":"lambda_symbol**2 / (kappa**2 * tau + lambda_symbol)**2"}

Original: \(\lambda \ge 1 + (\beta-\alpha)/2\)
Output: {"answer_kind":"single","sympy_answer":"Ge(lambda_symbol, 1 + (beta_symbol - alpha)/2)"}

Original: \(10 (2\beta/27)^{1/5} \int_0^1 |f'(x)|^{4/5}\,dx\)
Output: {"answer_kind":"single","sympy_answer":"10 * (2*beta_symbol/27)**Rational(1, 5) * Integral(Abs(Derivative(f(x), x))**Rational(4, 5), (x, 0, 1))"}

Original: \((x,y,z) = (\pm 47, \pm 2, \pm 7)\)
Output: {"answer_kind":"single","sympy_answer":"FiniteSet(Tuple(47, 2, 7), Tuple(47, 2, -7), Tuple(47, -2, 7), Tuple(47, -2, -7), Tuple(-47, 2, 7), Tuple(-47, 2, -7), Tuple(-47, -2, 7), Tuple(-47, -2, -7))"}

Original: \((x,y,z) = (1,\pm 20,\pm 7)\)
Output: {"answer_kind":"single","sympy_answer":"FiniteSet(Tuple(1, 20, 7), Tuple(1, 20, -7), Tuple(1, -20, 7), Tuple(1, -20, -7))"}

Original: \(\{(n,1,k,0),(1,0,k,1): k \ge 2\} \cup \{(0,2,k,1),(1,3,k,2): k \ge 3\} \cup \{(0,3,2,1),(0,3,2,2)\}\)
Output: {"answer_kind":"single","sympy_answer":"Union(ImageSet(Lambda(k, Tuple(n, 1, k, 0)), Range(2, oo)), ImageSet(Lambda(k, Tuple(1, 0, k, 1)), Range(2, oo)), ImageSet(Lambda(k, Tuple(0, 2, k, 1)), Range(3, oo)), ImageSet(Lambda(k, Tuple(1, 3, k, 2)), Range(3, oo)), FiniteSet(Tuple(0, 3, 2, 1), Tuple(0, 3, 2, 2)))"}
"""


class AnswerNormalizationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_kind: str = Field(description="Either 'single' or 'multi'.")
    sympy_answer: str | list[str] = Field(
        description=(
            "A SymPy-parsable string when answer_kind='single', or an ordered list of SymPy-parsable strings when answer_kind='multi'."
        ),
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> AnswerNormalizationPayload:
        if self.answer_kind not in {"single", "multi"}:
            raise ValueError("answer_kind must be 'single' or 'multi'.")
        if self.answer_kind == "single" and not isinstance(self.sympy_answer, str):
            raise ValueError("answer_kind='single' requires sympy_answer to be a string.")
        if self.answer_kind == "multi":
            if not isinstance(self.sympy_answer, list):
                raise ValueError("answer_kind='multi' requires sympy_answer to be a list of strings.")
            if len(self.sympy_answer) < 2:
                raise ValueError("answer_kind='multi' requires at least two sympy strings.")
        return self

    @model_validator(mode="after")
    def _validate_content(self) -> AnswerNormalizationPayload:
        values = [self.sympy_answer] if isinstance(self.sympy_answer, str) else self.sympy_answer
        for sympy_string in values:
            if not isinstance(sympy_string, str):
                raise ValueError(f"SymPy answer item must be a string: {sympy_string!r}")
            if any(marker in sympy_string for marker in ("$", "\\", "^")):
                raise ValueError(f"SymPy string contains disallowed characters: {sympy_string!r}")
            try:
                parse_expr(sympy_string, evaluate=False)
            except Exception as exc:
                raise ValueError(f"SymPy string is not parsable: {sympy_string!r}") from exc
        return self


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


class NormalizationError(ValueError):
    pass


def build_model() -> Any:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_retries": MAX_RETRIES,
        "temperature": TEMPERATURE,
    }
    return ChatOpenAI(**kwargs)


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            if isinstance(payload.get("problems"), list):
                rows = [row for row in payload["problems"] if isinstance(row, dict)]
            elif isinstance(payload.get("results"), list):
                rows = [row for row in payload["results"] if isinstance(row, dict)]
            else:
                rows = [payload]
        else:
            rows = []
    else:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)

    return rows if limit < 0 else rows[:limit]


def normalize_sympy_string(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip())
    if not text:
        raise ValueError("SymPy string is empty.")
    return text


def parse_sympy_string(value: str) -> Any:
    if any(marker in value for marker in ("$", "\\")):
        raise ValueError(f"SymPy string still contains LaTeX or markdown wrappers: {value!r}")
    if "^" in value:
        raise ValueError(f"Use '**' instead of '^' in SymPy strings: {value!r}")
    return parse_expr(value, evaluate=False)


def normalize_sympy_answer(value: str | list[str]) -> str | list[str]:
    if isinstance(value, str):
        text = normalize_sympy_string(value)
        parse_sympy_string(text)
        return text
    normalized_values: list[str] = []
    for raw_value in value:
        text = normalize_sympy_string(raw_value)
        parse_sympy_string(text)
        normalized_values.append(text)
    return normalized_values


def build_normalized_answer(answer_kind: str, sympy_answer: str | list[str]) -> str:
    if answer_kind == "single":
        if not isinstance(sympy_answer, str):
            raise ValueError("Single normalized answer requires sympy_answer to be a string.")
        return sympy_answer
    if not isinstance(sympy_answer, list):
        raise ValueError("Multi normalized answer requires sympy_answer to be a list.")
    return json.dumps(sympy_answer, ensure_ascii=False)


def usage_from_message(message: AIMessage | None) -> Usage:
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

    input_tokens = first_int("input_tokens", "prompt_tokens")
    output_tokens = first_int("output_tokens", "completion_tokens")
    total_tokens = first_int("total_tokens", "totalTokens")
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def is_retryable_error(exc: Exception) -> bool:
    retryable_class_names = {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "RemoteProtocolError",
    }
    current: BaseException | None = exc
    while current is not None:
        if current.__class__.__name__ in retryable_class_names:
            return True
        message = str(current).lower()
        if any(marker in message for marker in ("timed out", "timeout", "rate limit", "too many requests", "429")):
            return True
        current = current.__cause__ or current.__context__
    return False


def message_text(message: AIMessage) -> str:
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
        return "\n".join(parts).strip()
    return str(content)


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise NormalizationError(f"Model response does not contain a JSON object: {text[:400]!r}")
    return stripped[start : end + 1]


def invoke_normalizer(
    *,
    llm: Any,
    logger: ConsoleLogger,
    problem_id: str,
    question: str,
    original_answer: str,
) -> dict[str, Any]:
    messages = [
        SystemMessage(content=PROMPT_TEMPLATE),
        HumanMessage(content=(f"Problem id: {problem_id}\n\nQuestion:\n{question}\n\nOriginal answer:\n{original_answer}")),
    ]
    raw_message = llm.invoke(messages)
    if not isinstance(raw_message, AIMessage):
        raise TypeError("Model did not return an AIMessage.")

    raw_text = message_text(raw_message)
    payload_text = extract_json_object(raw_text)
    try:
        parsed_payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise NormalizationError(f"Model returned invalid JSON: {payload_text[:400]!r}") from exc
    if not isinstance(parsed_payload, Mapping):
        raise NormalizationError("Model returned JSON that is not an object.")

    try:
        parsed = AnswerNormalizationPayload(**dict(parsed_payload))
    except Exception as exc:
        raise NormalizationError(f"Normalized payload validation failed: {exc}") from exc

    sympy_answer = normalize_sympy_answer(parsed.sympy_answer)
    normalized_answer = build_normalized_answer(parsed.answer_kind, sympy_answer)
    usage = usage_from_message(raw_message)

    logger.log_model_call(
        agent_id="answer_normalizer",
        turn=1,
        model_name=MODEL,
        messages=[
            {"role": "system", "content": PROMPT_TEMPLATE},
            {
                "role": "user",
                "content": (f"Problem id: {problem_id}\n\nQuestion:\n{question}\n\nOriginal answer:\n{original_answer}"),
            },
        ],
        raw_response=raw_text,
        parsed_payload={
            "answer_kind": parsed.answer_kind,
            "sympy_answer": sympy_answer,
            "normalized_answer": normalized_answer,
        },
        token_usage={
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        },
    )

    return {
        "answer_kind": parsed.answer_kind,
        "sympy_answer": sympy_answer,
        "normalized_answer": normalized_answer,
        "token_usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


def write_output(path: Path, *, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize reference answers into SymPy-friendly strings.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--retry-backoff-sec", type=float, default=DEFAULT_RETRY_BACKOFF_SEC)
    parser.add_argument("--question-field", type=str, default="question")
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--id-field", type=str, default="id")
    args = parser.parse_args()

    logger = ConsoleLogger(mode="normalize_answers")
    logger.setup_logging()
    logger.start_run(
        metadata={
            "mode": "normalize_answers",
            "dataset_path": str(args.input),
            "output_dir": str(args.output_dir),
            "model": MODEL,
        }
    )

    llm = build_model()
    rows = load_rows(args.input, args.limit)
    logger.progress(f"[normalize_answers] loaded rows={len(rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime(FILE_TIMESTAMP_FORMAT)
    output_path = args.output_dir / f"normalized_answers_{timestamp}.json"

    results: list[dict[str, Any]] = []
    summary = {
        "model": MODEL,
        "rows": len(rows),
        "completed_rows": 0,
        "successful_rows": 0,
        "failed_rows": 0,
        "dataset_path": str(args.input),
        "output_file": str(output_path),
    }

    for index, row in enumerate(rows):
        problem_id = row.get(args.id_field)
        if not isinstance(problem_id, str):
            problem_id = f"row-{index + 1:05d}"
        question = str(row.get(args.question_field, ""))
        original_answer = str(row.get(args.answer_field, ""))

        logger.start_run(
            metadata={
                "agent_id": "answer_normalizer",
                "problem_id": problem_id,
                "question": question,
                "original_answer": original_answer,
            }
        )

        result_payload: dict[str, Any]
        error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                result = invoke_normalizer(
                    llm=llm,
                    logger=logger,
                    problem_id=problem_id,
                    question=question,
                    original_answer=original_answer,
                )
                result_payload = {
                    "id": problem_id,
                    "question": question,
                    "original_answer": original_answer,
                    "normalized_answer": result["normalized_answer"],
                    "answer_kind": result["answer_kind"],
                    "sympy_answer": result["sympy_answer"],
                    "token_usage": result["token_usage"],
                    "error": None,
                }
                results.append(result_payload)
                summary["successful_rows"] += 1
                logger.progress(f"[normalize_answers] problem_id={problem_id} normalized as {result['answer_kind']}")
                logger.finish_run(status="normalized")
                break
            except Exception as exc:
                error = exc
                logger.finish_run(status="failed")
                if attempt >= args.max_attempts or not is_retryable_error(exc):
                    result_payload = {
                        "id": problem_id,
                        "question": question,
                        "original_answer": original_answer,
                        "normalized_answer": "",
                        "answer_kind": "",
                        "sympy_answer": "",
                        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                    results.append(result_payload)
                    summary["failed_rows"] += 1
                    break
                logger.progress(
                    f"[normalize_answers] retrying problem_id={problem_id} "
                    f"attempt={attempt + 1}/{args.max_attempts} after {exc.__class__.__name__}"
                )
                time.sleep(args.retry_backoff_sec)
                logger.start_run(
                    metadata={
                        "agent_id": "answer_normalizer",
                        "problem_id": problem_id,
                        "question": question,
                        "original_answer": original_answer,
                        "problem_attempt": attempt + 1,
                    }
                )

        summary["completed_rows"] = index + 1
        summary["last_problem_id"] = problem_id
        write_output(output_path, summary=summary, results=results)

        if error is not None:
            logger.progress(f"[normalize_answers] problem_id={problem_id} failed: {error.__class__.__name__}: {error}")

        if index + 1 < len(rows) and args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    logger.log_artifact(
        name="normalized-answers",
        path=output_path,
        artifact_type="normalized-answers",
        metadata={"rows": len(rows), "dataset_path": str(args.input)},
    )
    logger.finish_run(status="completed")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
