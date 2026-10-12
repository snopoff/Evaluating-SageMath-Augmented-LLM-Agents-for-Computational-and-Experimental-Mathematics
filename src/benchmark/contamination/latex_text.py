"""Helpers for reading arXiv e-prints and locating a theorem inside LaTeX sources.

Everything here is pure text processing (no network) so it can be unit tested.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from dataclasses import dataclass

THEOREM_ENV_RE = re.compile(
    r"\\begin\{(theorem|thm|theo|mainthm|maintheorem|proposition|prop|lemma|lem|corollary|cor)\*?\}",
    re.IGNORECASE,
)
PROOF_BEGIN_RE = re.compile(r"\\begin\{(?:proof|pf|Proof)\}(\[[^\]]*\])?|\\proof\b(\[[^\]]*\])?")
LABEL_RE = re.compile(r"\\label\{[^}]*\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*")
NORMALFONT_RE = re.compile(r"\\normalfont\s*")
# Paragraphs end at blank lines, sectioning commands, or environment boundaries.
PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n|\n(?=\\(?:sub)*section\*?\{|\\begin\{document\}|\\end\{|\\maketitle)")
LEADING_ENV_RE = re.compile(r"^\s*\\begin\{[^}]*\}(\[[^\]]*\])?\s*")
SECTIONING_LINE_RE = re.compile(r"^\s*\\(?:(?:sub)*section\*?\{[^}]*\}|begin\{document\}|maketitle)\s*", re.MULTILINE)


# --------------------------------------------------------------------------- e-print unpacking


def extract_tex_sources(blob: bytes) -> str | None:
    """Return the concatenated ``.tex`` files of an arXiv e-print, main file first.

    arXiv serves either a gzipped tarball, a single gzipped ``.tex`` file, or a PDF
    (no sources).  ``None`` means no LaTeX could be recovered.
    """

    if blob[:4] == b"%PDF":
        return None
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
            files: list[tuple[str, str]] = []
            for member in archive.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".tex"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                files.append((member.name, handle.read().decode("utf-8", errors="replace")))
            return _join_tex_files(files) if files else None
    except tarfile.TarError:
        pass
    try:
        text = gzip.decompress(blob).decode("utf-8", errors="replace")
    except (OSError, EOFError):
        text = blob.decode("utf-8", errors="replace")
    return text if "\\begin{document}" in text or "\\section" in text else None


def _join_tex_files(files: list[tuple[str, str]]) -> str:
    main = [text for _, text in files if "\\begin{document}" in text]
    rest = [text for _, text in files if "\\begin{document}" not in text]
    return "\n\n".join(main + rest)


# --------------------------------------------------------------------------- cleaning


def strip_comments(source: str) -> str:
    return "\n".join(COMMENT_RE.sub("", line) for line in source.splitlines())


def clean_tex(text: str) -> str:
    """Remove labels/comments and collapse whitespace so prompts and targets are compact."""

    text = strip_comments(text)
    text = LABEL_RE.sub("", text)
    text = NORMALFONT_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _normalize_for_match(text: str) -> tuple[str, list[int]]:
    """Lower-case, drop whitespace/labels; return the normalized text and index map."""

    skip = [False] * len(text)
    for pattern in (LABEL_RE, NORMALFONT_RE):
        for match in pattern.finditer(text):
            for index in range(match.start(), match.end()):
                skip[index] = True
    chars: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate(text):
        if char.isspace() or skip[index]:
            continue
        chars.append(char.lower())
        index_map.append(index)
    return "".join(chars), index_map


# --------------------------------------------------------------------------- theorem context


@dataclass(frozen=True)
class TheoremContext:
    env_name: str
    statement: str  # cleaned theorem body
    pretext: str | None  # cleaned paragraph immediately before the environment
    proof_start: str | None  # cleaned opening of the proof that follows
    matched_by: str  # "field" (RealMath theorem text) or "first_env"


def locate_theorem(source: str, theorem_text: str, *, needle_chars: int = 60) -> tuple[int, int] | None:
    """Find the RealMath ``theorem`` field inside the LaTeX source; return char span."""

    source_norm, index_map = _normalize_for_match(source)
    needle_norm, _ = _normalize_for_match(clean_tex(theorem_text))
    if len(needle_norm) < 20:
        return None
    needle = needle_norm[:needle_chars]
    start = source_norm.find(needle)
    if start < 0:
        # Try a shorter needle: RealMath sometimes trims the first token(s).
        needle = needle_norm[10 : 10 + needle_chars]
        start = source_norm.find(needle)
        if start < 0:
            return None
    end = min(start + len(needle_norm), len(index_map) - 1)
    return index_map[start], index_map[end] + 1


def _enclosing_env(source: str, start: int, end: int) -> tuple[str, int, int] | None:
    """Return (env name, begin index, end index) of the theorem-like environment around a span."""

    window_start = max(0, start - 400)
    begins = [match for match in THEOREM_ENV_RE.finditer(source, window_start, end) if match.start() <= start]
    if not begins:
        return None
    begin = begins[-1]
    env = begin.group(1)
    # Search from ``start``: the matched span may overshoot the environment by a few
    # characters when RealMath's text and the source differ at the very end.
    end_match = re.compile(r"\\end\{" + re.escape(env) + r"\*?\}").search(source, start)
    if end_match is None:
        return None
    return env, begin.start(), end_match.end()


def _last_paragraph_before(source: str, index: int, *, max_chars: int = 2500, min_chars: int = 200) -> str | None:
    """The closest preceding paragraph that is real prose (long enough, mostly letters)."""

    before = source[max(0, index - max_chars) : index]
    for raw in reversed(PARAGRAPH_BREAK_RE.split(before)):
        paragraph = clean_tex(LEADING_ENV_RE.sub("", SECTIONING_LINE_RE.sub("", raw)))
        if len(paragraph) < min_chars:
            continue
        letters = sum(1 for char in paragraph if char.isalpha())
        if letters / len(paragraph) < 0.55:
            continue
        return paragraph
    return None


def _proof_after(source: str, index: int, *, lookahead: int = 1500, max_chars: int = 500) -> str | None:
    match = PROOF_BEGIN_RE.search(source, index, index + lookahead)
    if match is None:
        return None
    body = source[match.end() : match.end() + max_chars * 3]
    end = min((pos for pos in (body.find(marker) for marker in ("\\end{proof}", "\\end{pf}", "\\end{Proof}", "\\endproof", "\\qed")) if pos >= 0), default=-1)
    if end >= 0:
        body = body[:end]
    body = clean_tex(body)
    return body[:max_chars] if len(body) >= 80 else None


def theorem_context(source: str, theorem_text: str | None, *, max_statement_chars: int = 3000) -> TheoremContext | None:
    """Locate the theorem (by RealMath text, else the first theorem-like env) and its context."""

    source = strip_comments(source)
    body_start = source.find("\\begin{document}")
    if body_start < 0:
        body_start = 0

    env_span: tuple[str, int, int] | None = None
    matched_by = "field"
    if theorem_text:
        span = locate_theorem(source, theorem_text)
        if span is not None:
            env_span = _enclosing_env(source, *span)
    if env_span is None:
        matched_by = "first_env"
        begin = THEOREM_ENV_RE.search(source, body_start)
        if begin is None:
            return None
        env_span = _enclosing_env(source, begin.start(), begin.end())
        if env_span is None:
            return None

    env, begin_index, end_index = env_span
    inner = re.sub(r"^\\begin\{[^}]*\}(\[[^\]]*\])?", "", source[begin_index:end_index])
    inner = re.sub(r"\\end\{[^}]*\}$", "", inner)
    # A statement that runs into a proof means the environment end was not where
    # expected (unbalanced sources); keep the statement part only.
    inner = PROOF_BEGIN_RE.split(inner)[0]
    statement = clean_tex(inner)
    if len(statement) < 40 or len(statement) > max_statement_chars:
        return None
    return TheoremContext(
        env_name=env,
        statement=statement,
        pretext=_last_paragraph_before(source, begin_index),
        proof_start=_proof_after(source, end_index),
        matched_by=matched_by,
    )
