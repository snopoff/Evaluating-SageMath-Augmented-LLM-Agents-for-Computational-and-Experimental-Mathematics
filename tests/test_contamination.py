"""Offline tests for the contamination-probe pipeline (no network, no API)."""

from __future__ import annotations

import gzip
import io
import json
import tarfile

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.benchmark.contamination import probes as probes_mod
from src.benchmark.contamination.corpus import allocate, parse_atom
from src.benchmark.contamination.latex_text import extract_tex_sources, locate_theorem, theorem_context
from src.benchmark.contamination.scoring import (
    longest_run,
    ngram_hit_rate,
    parse_instant_answer,
    parse_quiz_letter,
    rouge_l,
    score_continuation,
)

SOURCE = r"""
\documentclass{article}
\newtheorem{theorem}{Theorem}
\begin{document}
\section{Introduction}
Harmonic centrality was introduced as a robust alternative to closeness. In this paper we study the
direct product of a path and a cycle, which arises naturally in the analysis of circular ladders. Our
main tool is a distance decomposition lemma that reduces the computation to a sum of harmonic numbers.
The result below is the first of several closed forms. % trailing comment

\begin{theorem}\label{thm:main}
\normalfont For the direct product of path $P_2$ of order 2 and a cycle graph $C_m$ of order $m$, the
harmonic centrality of any vertex $(u_i, v_j)$ is given by $\frac{1}{2m-1}(2H_{m-1}+\frac{1}{m})$ if $m$ is odd.
\end{theorem}
\begin{proof}
Fix a vertex $(u_i, v_j)$. By the distance decomposition lemma, every other vertex lies at distance equal to its
cyclic distance from $v_j$ when the parities agree, and one more otherwise. Summing reciprocals gives the claim.
\end{proof}
\end{document}
"""

THEOREM_FIELD = (
    "\\normalfont \n\tFor the direct product of path $P_2$ of order 2 and a cycle graph $C_m$ of order $m$, the "
    "harmonic centrality of any vertex $(u_i, v_j)$ is given by"
)


def test_locate_theorem_and_context() -> None:
    assert locate_theorem(SOURCE, THEOREM_FIELD) is not None
    context = theorem_context(SOURCE, THEOREM_FIELD)
    assert context is not None
    assert context.env_name == "theorem"
    assert context.matched_by == "field"
    assert context.statement.startswith("For the direct product")
    assert "\\label" not in context.statement
    assert context.pretext is not None and context.pretext.startswith("Harmonic centrality")
    assert "trailing comment" not in context.pretext
    assert context.proof_start is not None and context.proof_start.startswith("Fix a vertex")


def test_first_env_fallback_for_control_papers() -> None:
    context = theorem_context(SOURCE, None)
    assert context is not None and context.matched_by == "first_env"


def test_extract_tex_sources_from_tarball_and_pdf() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in (("macros.tex", "\\def\\R{\\mathbb R}"), ("main.tex", SOURCE)):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    text = extract_tex_sources(buffer.getvalue())
    assert text is not None and text.startswith("\n\\documentclass")  # main file first
    assert extract_tex_sources(b"%PDF-1.5 ...") is None
    assert extract_tex_sources(gzip.compress(SOURCE.encode())) is not None


def test_parse_atom() -> None:
    xml = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    <entry><id>http://arxiv.org/abs/2205.03791v2</id><title>A  title\n here</title><summary>Sum mary.</summary>
    <published>2022-05-08T12:00:00Z</published><author><name>A. B.</name></author>
    <arxiv:primary_category term="math.CO"/></entry></feed>"""
    entries = parse_atom(xml)
    assert entries == [
        {"arxiv_id": "2205.03791", "title": "A title here", "abstract": "Sum mary.", "primary_category": "math.CO", "published": "2022-05-08", "authors": ["A. B."]}
    ]


def test_allocate_largest_remainder() -> None:
    assert allocate({"a": 5, "b": 3, "c": 2}, 4) == {"a": 2, "b": 1, "c": 1}
    assert sum(allocate({"a": 1, "b": 1, "c": 1}, 10).values()) == 10


def test_scoring_metrics() -> None:
    target = "we prove that every finite group of odd order is solvable using character theory"
    assert rouge_l(target, target) == 1.0
    assert rouge_l("completely unrelated words here", target) < 0.2
    assert ngram_hit_rate(target, target) == 1.0
    assert longest_run("every finite group of odd order is solvable", target) == 8
    score = score_continuation("we prove that every finite group", target)
    assert 0 < score.rouge_l < 1 and score.longest_run == 6


def test_parsers() -> None:
    assert parse_quiz_letter(" (C) ") == "C"
    assert parse_quiz_letter("Answer: e") == "E"
    assert parse_quiz_letter("no idea") is None
    parsed = parse_instant_answer('```json\n{"final_answer": "$2^n$", "sympy_answer": "2**n"}\n```')
    assert parsed == {"final_answer": "$2^n$", "sympy_answer": "2**n", "parsed": True}
    assert parse_instant_answer("NONE")["parsed"] is False


def test_quiz_variants_are_distinct_and_deterministic() -> None:
    question = "Let $G$ be a finite group. What is the number of Sylow subgroups such that the order divides 12? Provide the count."
    variants = probes_mod.quiz_variants(question, seed=7)
    assert variants is not None and len(variants) == 3 and len(set(variants)) == 3
    assert question not in variants
    assert variants == probes_mod.quiz_variants(question, seed=7)


def test_build_probes_from_corpus() -> None:
    abstract = " ".join(f"Sentence number {i} says something about groups and rings." for i in range(8))
    corpus = {
        "papers": [
            {
                "arxiv_id": "2205.00001",
                "group": "benchmark",
                "cohort": "2022",
                "title": "T",
                "published": "2022-05-01",
                "abstract": abstract,
                "problem_ids": ["realmathbenchmark-00001"],
                "pretext": " ".join(f"Context sentence {i} explains the setting in detail." for i in range(8)),
                "theorem_statement": "For every $n$ the invariant equals $n^2$ and the bound is sharp for all cases considered.",
                "proof_start": "Fix $n$. " * 40,
                "theorem_env": "theorem",
            },
            {"arxiv_id": "2509.00001", "group": "fresh", "cohort": None, "title": "F", "published": "2026-09-01", "abstract": abstract, "problem_ids": []},
        ]
    }
    problems = {
        "realmathbenchmark-00001": {
            "id": "realmathbenchmark-00001",
            "question": "Let $G$ be a group. What is the order of $G$ such that it is simple? Provide the value.",
            "original_answer": "60",
            "sympy_answer": "60",
            "answer_kind": "single",
        }
    }
    records = probes_mod.build_probes(corpus, problems, probes=list(probes_mod.PROBES), seed=0)
    kinds = sorted((r.probe, r.mode, r.group) for r in records)
    assert kinds == [
        ("abstract", "guided", "benchmark"),
        ("abstract", "guided", "fresh"),
        ("abstract", "unguided", "benchmark"),
        ("abstract", "unguided", "fresh"),
        ("instant", "none", "benchmark"),
        ("pretext", "guided", "benchmark"),
        ("pretext", "unguided", "benchmark"),
        ("proof", "guided", "benchmark"),
        ("quiz", "none", "benchmark"),
    ]
    quiz = next(r for r in records if r.probe == "quiz")
    assert quiz.target in "ABCD" and quiz.meta["options"]["ABCD".index(quiz.target)] == problems["realmathbenchmark-00001"]["question"]
    for unguided in (r for r in records if r.mode == "unguided"):
        assert "arXiv" not in unguided.user
    json.dumps([r.__dict__ for r in records])  # serialisable


def test_abstract_short_and_single_sentence_papers_are_retained() -> None:
    abstracts = [
        "We prove a sharp bound for the invariant of every finite group considered here",
        "We study finite groups. The bound is sharp.",
    ]
    for group in ("benchmark", "famous", "same_vintage", "fresh"):
        for abstract in abstracts:
            paper = {"arxiv_id": "test", "title": "Test", "group": group, "abstract": abstract}
            records = probes_mod.abstract_records(paper)
            assert {r.mode for r in records} == {"guided", "unguided"}
            assert records[0].target == records[1].target
            for record in records:
                assert record.meta["split_method"] == "word_fallback"
                prefix = record.user.split("Beginning of the abstract:\n\n")[1].split("\n\nContinuation:")[0]
                assert prefix and record.target
                assert prefix + " " + record.target == abstract
                assert record.target not in record.user


def test_existing_abstract_split_is_preserved() -> None:
    abstract = " ".join(f"Sentence number {i} describes a mathematical construction in detail." for i in range(8))
    expected = probes_mod.split_prefix(abstract, probes_mod.ABSTRACT_PREFIX_FRACTION)
    assert expected is not None
    records = probes_mod.abstract_records({"arxiv_id": "test", "title": "Test", "group": "famous", "abstract": abstract})
    for record in records:
        assert record.meta["split_method"] == "sentence"
        assert record.target == expected[1]
        assert "Beginning of the abstract:\n\n" + expected[0] + "\n\nContinuation:" in record.user
