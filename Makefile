.PHONY: reproduce test test-analysis chat chat-tool chat-no-tool generate-predictions generate-predictions-tool generate-predictions-no-tool benchmark judge-sympy
reproduce:
	uv run --frozen --extra analysis python scripts/paper_revision_analysis.py
	uv run --frozen --extra analysis python scripts/additional_statistical_analysis.py
	uv run --frozen --extra analysis python scripts/construction_analysis.py
	uv run --frozen --extra analysis python -m src.benchmark.leave_one_judge_out
	uv run --frozen --extra analysis python -m src.benchmark.contamination.report --from-scores

test test-analysis:
	uv run --frozen --extra analysis python -m pytest $(PYTEST)

chat chat-tool:
	uv run --frozen python main.py $(HYDRA)
chat-no-tool:
	uv run --frozen python main.py system_prompt=no-tool '~sage_skill' 'tools=[]' $(HYDRA)
generate-predictions generate-predictions-tool:
	uv run --frozen python main.py --config-name generate_predictions tool_mode=tool $(HYDRA)
generate-predictions-no-tool:
	uv run --frozen python main.py --config-name generate_predictions tool_mode=no_tool $(HYDRA)
benchmark:
	uv run --frozen python benchmark.py $(HYDRA)
judge-sympy:
	@test -n "$(input)" || (echo "Set input=<saved-predictions-file>"; exit 1)
	uv run --frozen python src/benchmark/judge_sympy_answers.py --input "$(input)" --judge openai=gpt-5.5 --judge anthropic=claude-opus-4-7 --judge google=gemini-3.5-flash --resume
