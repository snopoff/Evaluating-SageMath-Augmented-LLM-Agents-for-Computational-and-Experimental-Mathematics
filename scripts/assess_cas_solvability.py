from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LITELLM_API_KEY = os.environ["LITELLM_API_KEY"]
LITELLM_BASE_URL = os.environ["LITELLM_BASE_URL"]
MODEL = "claude-opus-4-7"
MAX_TOKENS = 36024

CONTEXT7_API_KEY = os.environ.get("CONTEXT7_API_KEY", "")
CONTEXT7_MCP_URL = "https://mcp.context7.com/mcp"
CONTEXT7_TIMEOUT_SEC = 60
CONTEXT7_DEFAULT_TOKENS = 2000
CONTEXT7_DEFAULT_LIBRARY_ID = "/sagemath/documentation"
CONTEXT7_USE_API_KEY = os.environ.get("CONTEXT7_USE_API_KEY", "").lower() in {"1", "true", "yes"}

MAX_TOOL_ITERATIONS = 100

DEFAULT_INPUT_PATH = Path("realmathbenchmark_full.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/realmathbenchmark_full_cas_assessment.json")
DEFAULT_LIMIT = 300

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "cas_solvability_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "can_solve_with_cas": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": (
                        "Whether SageMath is useful in the research workflow for this problem — "
                        "for validating claims, filtering hypotheses, searching for counterexamples, "
                        "or computing small cases. Not whether SageMath directly produces the final answer."
                    ),
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Confidence in the verdict on a 1 to 5 scale.",
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief justification for the verdict (one or two sentences).",
                },
                "sagemath_modules": {
                    "type": "array",
                    "description": (
                        "If can_solve_with_cas is 'yes', list ONLY the names of relevant SageMath "
                        "modules / classes / functions that could help — for building the objects, "
                        "computing invariants, validating claims, or searching counterexamples (e.g. "
                        "'sage.graphs.graph.Graph', 'Graph.cartesian_product', 'PolynomialRing', "
                        "'GF', 'sage.matrix.matrix_space.MatrixSpace'). Names only. No implementation "
                        "details, no code, no step-by-step instructions. A module that only covers a "
                        "building block still counts. If 'no', leave this list empty."
                    ),
                    "items": {"type": "string"},
                },
                "missing_capabilities": {
                    "type": "string",
                    "description": (
                        "If can_solve_with_cas is 'no', describe what role Sage would need to play "
                        "for this problem (e.g. 'construct objects of type X and compute invariant Y') "
                        "and state explicitly that no SageMath module / class / method covers it. "
                        "If 'yes', leave empty."
                    ),
                },
            },
            "required": [
                "can_solve_with_cas",
                "confidence",
                "explanation",
                "sagemath_modules",
                "missing_capabilities",
            ],
            "additionalProperties": False,
        },
    },
}

GET_LIBRARY_DOCS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_library_docs",
        "description": (
            "Fetch up-to-date documentation snippets for a Context7-compatible library, focused on "
            "a specific topic. Use this to verify which SageMath modules / classes / functions can "
            "solve the problem, or to confirm that no module covers a needed capability. "
            f"For SageMath, libraryID is '{CONTEXT7_DEFAULT_LIBRARY_ID}'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "context7CompatibleLibraryID": {
                    "type": "string",
                    "description": (
                        "Context7-compatible library ID in the form '/owner/repo' "
                        f"(e.g. '{CONTEXT7_DEFAULT_LIBRARY_ID}')."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Topic, mathematical concept, or keyword to focus the docs on "
                        "(e.g. 'graph cartesian product', 'PolynomialRing factorization')."
                    ),
                },
                "tokens": {
                    "type": "integer",
                    "minimum": 500,
                    "maximum": 8000,
                    "description": "Approximate maximum number of tokens of documentation to return.",
                },
            },
            "required": ["context7CompatibleLibraryID", "topic"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_ASSESSMENT_TOOL_NAME = "submit_cas_assessment"

SUBMIT_ASSESSMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBMIT_ASSESSMENT_TOOL_NAME,
        "description": "Submit the final CAS solvability assessment. This is the only valid final answer.",
        "strict": True,
        "parameters": RESPONSE_SCHEMA["json_schema"]["schema"],
    },
}

TOOLS: list[dict[str, Any]] = [
    GET_LIBRARY_DOCS_TOOL,
    {
        **SUBMIT_ASSESSMENT_TOOL,
    },
]

PROMPT_TEMPLATE = """You assess whether a mathematics problem can be tackled by an LLM-driven research workflow that uses a computer algebra system (CAS) such as SageMath as an assistant.

The question is whether SageMath would be USEFUL in the research loop — for any of:
- computing concrete small cases / examples,
- evaluating numerical or symbolic invariants on those cases,
- validating a candidate formula or claim on multiple instances,
- filtering competing hypotheses by checking which one survives,
- searching for counterexamples to a conjecture,
- partial symbolic manipulation that supports the human/LLM reasoning.

If SageMath helps with at least one of these roles for this problem, the verdict is "yes" — even if the final answer (a parametric formula, a proof) still has to be produced by a human or LLM and Sage only validates / filters / refutes.

You have one tool backed by the Context7 MCP server, which serves up-to-date SageMath documentation:
- get_library_docs(context7CompatibleLibraryID, topic, tokens) -- fetches focused documentation snippets for a given topic. For SageMath, libraryID is '{library_id}'.

You also have one final-answer tool:
- submit_cas_assessment(...) -- submit the final structured assessment. You MUST use this tool for the final answer. Do not put the final JSON in plain message text.

Recommended workflow ("use context7"):
- For each mathematical concept that appears in the problem (e.g. graph products, harmonic centrality, polynomial factorization, ideal reduction, spectra of Laplacians, Hankel matrices over finite fields), call get_library_docs with libraryID='{library_id}' and a focused topic.
- You MUST call get_library_docs at least once before finalizing the answer.
- You are NOT looking for a method that directly produces the final answer. You are looking for ANY module / class / function that could plausibly help build the relevant objects, run experiments, or check claims — even if it only covers a building block (e.g. a graph constructor, an eigenvalue routine, a polynomial ring, an ideal reducer, a finite-field type).
- Ground your verdict and the sagemath_modules list in the snippets you actually saw. Do not invent module paths.

Verdict rules (must follow strictly):
- can_solve_with_cas = "yes" if SageMath provides at least a relevant module / class / method that could help validate, filter, refute, or experiment with the problem — even partially. List those modules / classes / functions in sagemath_modules (names only, no implementation details, no code, no step-by-step instructions). Leave missing_capabilities = "".
- can_solve_with_cas = "no" only if SageMath has nothing usable for this problem at all — no relevant constructor, no relevant invariant, no relevant algebraic structure. Leave sagemath_modules = []. In missing_capabilities describe what role Sage would need to play (e.g. "construct objects of type X and compute invariant Y") and state explicitly that no SageMath module / class / method covers it.

Submit the final assessment with submit_cas_assessment using these fields:
- can_solve_with_cas: "yes" or "no"
- confidence: integer 1-5 (1 = pure guess, 5 = certain)
- explanation: one or two short sentences. Describe HOW Sage would be used (validation / hypothesis filtering / counterexample search / small-case computation), not how to derive the final answer.
- sagemath_modules: list of relevant SageMath modules / classes / functions if "yes" (names only); empty list if "no"
- missing_capabilities: if "no", the role Sage would need to play and that no SageMath module covers it; empty string if "yes"

Problem id: {problem_id}

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


_litellm = OpenAI(
    api_key=LITELLM_API_KEY,
    base_url=LITELLM_BASE_URL,
)


class MCPError(RuntimeError):
    pass


class MCPClient:
    """Minimal Streamable-HTTP MCP client. Handles initialize, notifications, tools/call."""

    PROTOCOL_VERSION = "2025-03-26"

    def __init__(self, url: str, api_key: str = "", timeout_sec: int = 60):
        self.url = url
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.session_id: str | None = None
        self._next_id = 1
        self._initialized = False

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "llmxcas-cas-assessor/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["CONTEXT7_API_KEY"] = self.api_key
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    @staticmethod
    def _parse_response(body: str, content_type: str) -> dict[str, Any]:
        body = body.strip()
        if not body:
            return {}
        if "text/event-stream" in content_type or body.startswith("event:") or body.startswith("data:"):
            for raw_line in body.splitlines():
                line = raw_line.strip()
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        return json.loads(data)
                    except json.JSONDecodeError:
                        continue
            raise MCPError(f"no JSON-RPC frame in SSE payload: {body[:200]!r}")
        return json.loads(body)

    def _post(self, payload: dict[str, Any], expect_response: bool) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                if not self.session_id:
                    sid = response.headers.get("Mcp-Session-Id")
                    if sid:
                        self.session_id = sid
                content_type = response.headers.get("Content-Type", "")
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                err_body = ""
            raise MCPError(f"HTTP {exc.code} {exc.reason}: {err_body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise MCPError(f"network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise MCPError("timeout") from exc

        if not expect_response:
            return {}
        return self._parse_response(body, content_type)

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        response = self._post(payload, expect_response=True)
        if "error" in response:
            err = response["error"]
            raise MCPError(f"JSON-RPC error: {err.get('code')} {err.get('message')} {err.get('data', '')}")
        return response.get("result", {})

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._post(payload, expect_response=False)

    def initialize(self) -> None:
        if self._initialized:
            return
        self._request(
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "llmxcas-cas-assessor", "version": "1.0"},
            },
        )
        self._notify("notifications/initialized")
        self._initialized = True

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.initialize()
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            content = result.get("content") or []
            text = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
            raise MCPError(f"tool '{name}' returned error: {text[:500]}")
        content = result.get("content") or []
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts).strip() or json.dumps(result, ensure_ascii=False)


_mcp_client = MCPClient(
    url=CONTEXT7_MCP_URL,
    api_key=CONTEXT7_API_KEY if CONTEXT7_USE_API_KEY else "",
    timeout_sec=CONTEXT7_TIMEOUT_SEC,
)



def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    if name != "get_library_docs":
        return f"[tool error] unknown tool: {name}"

    library_id = str(arguments.get("context7CompatibleLibraryID", "")).strip()
    topic = str(arguments.get("topic", "")).strip()
    if not library_id:
        return "[tool error] missing 'context7CompatibleLibraryID'"
    if not topic:
        return "[tool error] missing 'topic'"
    mcp_args: dict[str, Any] = {
        "libraryId": library_id,
        "query": topic,
        "researchMode": False,
    }

    try:
        return _mcp_client.call_tool("query-docs", mcp_args)
    except MCPError as exc:
        if "Tool query-docs not found" in str(exc):
            legacy_args: dict[str, Any] = {
                "context7CompatibleLibraryID": library_id,
                "topic": topic,
            }
            tokens = arguments.get("tokens")
            if tokens is not None:
                legacy_args["tokens"] = max(500, min(int(tokens), 8000))
            try:
                return _mcp_client.call_tool("get-library-docs", legacy_args)
            except MCPError as legacy_exc:
                return f"[context7 mcp error] {legacy_exc}"
        return f"[context7 mcp error] {exc}"



def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        serialized.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        )
    return serialized


def _call_litellm_loop(
    prompt: str,
    model: str,
    max_tokens: int,
    tools: list[dict[str, Any]],
    *,
    on_tool_call: Any | None = None,
) -> tuple[str, Usage, list[dict[str, Any]]]:
    """Run a tool-calling loop until the model submits the final assessment tool."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_trace: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for iteration in range(MAX_TOOL_ITERATIONS):
        docs_called = any(entry.get("name") == "get_library_docs" for entry in tool_trace)
        forced_tool_name = SUBMIT_ASSESSMENT_TOOL_NAME if docs_called else "get_library_docs"
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": forced_tool_name}},
        )

        response = _litellm.chat.completions.create(**kwargs)
        usage = response.usage
        if usage is not None:
            total_prompt_tokens += usage.prompt_tokens or 0
            total_completion_tokens += usage.completion_tokens or 0

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Use the {forced_tool_name} tool. Do not answer in plain text; "
                        "the final assessment must be submitted as tool arguments."
                    ),
                }
            )
            continue

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": _serialize_tool_calls(tool_calls),
            }
        )

        for tc in tool_calls:
            if tc.function.name == SUBMIT_ASSESSMENT_TOOL_NAME:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    tool_output = f"[tool error] invalid final JSON arguments: {exc}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_output,
                        }
                    )
                    continue
                return (
                    json.dumps(arguments, ensure_ascii=False),
                    Usage(total_prompt_tokens, total_completion_tokens),
                    tool_trace,
                )

            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                tool_output = f"[tool error] invalid JSON arguments: {exc}"
                arguments = {}
            else:
                tool_output = execute_tool(tc.function.name, arguments)

            preview = tool_output if len(tool_output) <= 8000 else tool_output[:8000] + "\n[...truncated...]"
            trace_entry = {
                "iteration": iteration + 1,
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "arguments": arguments,
                "output_preview": preview[:500],
                "output_length": len(tool_output),
            }
            tool_trace.append(trace_entry)
            if on_tool_call is not None:
                on_tool_call(trace_entry)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": preview,
                }
            )

    return (
        "",
        Usage(total_prompt_tokens, total_completion_tokens),
        tool_trace,
    )



def load_problems(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(problem: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        problem_id=problem.get("id", ""),
        theorem=str(problem.get("theorem", "")).strip(),
        question=str(problem.get("question", "")).strip(),
        answer=str(problem.get("answer", "")).strip(),
        library_id=CONTEXT7_DEFAULT_LIBRARY_ID,
    )


def parse_assessment(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"_parse_error": f"JSONDecodeError: {exc}", "_raw": raw}
    if not isinstance(payload, dict):
        return {"_parse_error": "top-level not an object", "_raw": raw}
    return payload


def assess_problem(problem: dict[str, Any]) -> dict[str, Any]:
    prompt = build_prompt(problem)
    started = time.perf_counter()

    def _on_tool_call(entry: dict[str, Any]) -> None:
        args = entry.get("arguments") or {}
        if entry["name"] == "get_library_docs":
            label = f"libraryID={args.get('context7CompatibleLibraryID')!r}, topic={args.get('topic')!r}"
        else:
            label = json.dumps(args, ensure_ascii=False)
        print(
            f"  [mcp] {entry['name']}({label}) -> {entry['output_length']} chars",
            flush=True,
        )
        preview = _truncate(entry.get("output_preview") or "")
        if preview:
            indented = "\n".join("    " + line for line in preview.splitlines())
            print(indented, flush=True)

    raw, usage, tool_trace = _call_litellm_loop(
        prompt=prompt,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS,
        on_tool_call=_on_tool_call,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    parsed = parse_assessment(raw or "")

    return {
        "id": problem.get("id"),
        "question": problem.get("question"),
        "theorem": problem.get("theorem"),
        "answer": problem.get("answer"),
        "llm": {
            "model": MODEL,
            "can_solve_with_cas": parsed.get("can_solve_with_cas"),
            "confidence": parsed.get("confidence"),
            "explanation": parsed.get("explanation"),
            "sagemath_modules": parsed.get("sagemath_modules"),
            "missing_capabilities": parsed.get("missing_capabilities"),
            "raw_response": raw,
            "parse_error": parsed.get("_parse_error"),
            "tool_calls": tool_trace,
            "tool_call_count": len(tool_trace),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "elapsed_ms": elapsed_ms,
        },
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
    parser = argparse.ArgumentParser(description="Assess CAS-solvability of dataset problems with Opus 4.7 via LiteLLM + Context7 MCP.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to the input dataset JSON / JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write the assessment JSON.")
    args = parser.parse_args()

    problems = load_problems(args.input)
    if DEFAULT_LIMIT and DEFAULT_LIMIT > 0:
        problems = problems[:DEFAULT_LIMIT]

    print(f"Loaded {len(problems)} problems from {args.input}", flush=True)

    results: list[dict[str, Any]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "yes": 0, "no": 0, "errors": 0, "tool_calls": 0}

    separator = "_" * 80

    for index, problem in enumerate(problems, start=1):
        problem_id = problem.get("id", f"row-{index:05d}")
        theorem = str(problem.get("theorem", "")).strip()
        question = str(problem.get("question", "")).strip()
        reference_answer = str(problem.get("answer", "")).strip()

        print(separator, flush=True)
        print(f"Problem [{index}/{len(problems)}]  id={problem_id}", flush=True)
        print(f"\nTheorem:\n{_truncate(theorem)}", flush=True)
        print(f"\nQuestion:\n{_truncate(question)}", flush=True)
        print(f"\nReference answer:\n{_truncate(reference_answer)}", flush=True)

        try:
            record = assess_problem(problem)
        except Exception as exc:  # noqa: BLE001 — keep going on a single failure
            print(f"\nLLM answer:\n  ERROR: {_truncate(exc)}", flush=True)
            record = {
                "id": problem_id,
                "question": problem.get("question"),
                "theorem": problem.get("theorem"),
                "answer": problem.get("answer"),
                "llm": {"model": MODEL, "error": str(exc)},
            }
            totals["errors"] += 1
        else:
            llm = record["llm"]
            totals["prompt_tokens"] += llm.get("prompt_tokens") or 0
            totals["completion_tokens"] += llm.get("completion_tokens") or 0
            totals["tool_calls"] += llm.get("tool_call_count") or 0
            verdict = llm.get("can_solve_with_cas")
            if verdict in totals:
                totals[verdict] += 1
            modules = llm.get("sagemath_modules") or []
            missing = (llm.get("missing_capabilities") or "").strip()
            print("\nLLM answer:", flush=True)
            print(f"  can_solve_with_cas: {_truncate(verdict)}", flush=True)
            print(f"  confidence: {llm.get('confidence')}", flush=True)
            print(f"  explanation: {_truncate(llm.get('explanation'))}", flush=True)
            if modules:
                print(f"  sagemath_modules ({len(modules)}):", flush=True)
                for module in modules:
                    print(f"    - {_truncate(module)}", flush=True)
            else:
                print("  sagemath_modules: (none)", flush=True)
            if missing:
                print("  missing_capabilities:", flush=True)
                print(f"    {_truncate(missing)}", flush=True)
            else:
                print("  missing_capabilities: (none)", flush=True)
            print(
                f"  (tool_calls={llm.get('tool_call_count')}, "
                f"input_tokens={llm.get('prompt_tokens')}, output_tokens={llm.get('completion_tokens')}, "
                f"{llm.get('elapsed_ms')} ms)",
                flush=True,
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

    print(
        f"Done. yes={totals['yes']} no={totals['no']} errors={totals['errors']} "
        f"tool_calls={totals['tool_calls']} "
        f"prompt_tokens={totals['prompt_tokens']} completion_tokens={totals['completion_tokens']}",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
