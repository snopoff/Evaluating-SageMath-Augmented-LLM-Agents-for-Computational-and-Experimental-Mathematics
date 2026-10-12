"""Black-box contamination probes for the RealMath source papers .

Pipeline (each step is a CLI, see module docstrings):

1. ``corpus.py``   - collect benchmark papers plus control groups and cache arXiv
                     abstracts / LaTeX sources under ``data/reviews/contamination``.
2. ``probes.py``   - turn the corpus into prompt/target records (``probes.jsonl``).
3. ``run.py``      - query the models, one response file per model.
4. ``report.py``   - score responses and write ``report.md`` / ``report.json``.
"""
