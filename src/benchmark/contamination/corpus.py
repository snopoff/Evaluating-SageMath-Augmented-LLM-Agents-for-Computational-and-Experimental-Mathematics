"""Collect the papers used by the contamination probes and cache their arXiv data.

Four groups of papers are assembled:

* ``benchmark``     - the 88 source papers of the 133 evaluated problems.
* ``same_vintage``  - random math papers from the same submission windows
                      (May-June 2022, December 2024) and the same primary categories,
                      not in RealMath.  Tells whether the benchmark papers are recalled
                      better than an ordinary paper of the same age.
* ``fresh``         - math papers submitted in the last ``--fresh-window-days`` days,
                      same category mix.  Models cannot have seen them: the floor.
* ``famous``        - heavily reproduced papers (Attention, ResNet, ...) plus a few
                      celebrated math preprints: the ceiling and a sanity check that the
                      probes work at all.

Metadata comes from the arXiv API, sources from ``arxiv.org/e-print``; everything is
cached under ``data/reviews/contamination/cache`` so re-runs are free.  arXiv asks for
one request every three seconds, which is why fetching ~180 papers takes ~10 minutes.

Usage::

    uv run python src/benchmark/contamination/corpus.py            # all groups
    uv run python src/benchmark/contamination/corpus.py --groups benchmark famous --no-eprints
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import rootutils

ROOT = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.benchmark.contamination.latex_text import extract_tex_sources, theorem_context  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "data/reviews/contamination"
DEFAULT_PROBLEMS = ROOT / "data/processed/problems_to_run.json"
DEFAULT_REALMATH = ROOT / "data/construction/source_metadata.json"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_EPRINT = "https://arxiv.org/e-print/"
ARXIV_DELAY_SEC = 3.0
USER_AGENT = "anonymous-reproducibility-artifact/1.0"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

GROUPS = ("benchmark", "same_vintage", "fresh", "famous")

# Submission windows of the benchmark cohorts.
COHORT_WINDOWS: dict[str, tuple[date, date]] = {
    "2022": (date(2022, 5, 1), date(2022, 6, 30)),
    "2024": (date(2024, 12, 1), date(2024, 12, 31)),
}

# Ceiling group.  ML classics are memorised through millions of copies; the math
# preprints are famous but have far fewer copies, which is closer to our papers.
FAMOUS_ARXIV_IDS: tuple[str, ...] = (
    # machine learning classics
    "1706.03762",  # Attention is all you need
    "1512.03385",  # ResNet
    "1412.6980",  # Adam
    "1810.04805",  # BERT
    "1406.2661",  # GAN
    "1301.3781",  # word2vec
    "2005.14165",  # GPT-3
    "1707.06347",  # PPO
    "1409.1556",  # VGG
    "1312.6114",  # VAE
    "1502.03167",  # Batch normalization
    "1609.02907",  # GCN
    "1511.06434",  # DCGAN
    "1703.10593",  # CycleGAN
    "1503.02531",  # Knowledge distillation
    "1608.06993",  # DenseNet
    "1409.3215",  # Sequence to sequence
    "1409.0473",  # Bahdanau attention
    "1907.11692",  # RoBERTa
    "1910.10683",  # T5
    "2010.11929",  # Vision Transformer
    "2103.00020",  # CLIP
    "2006.11239",  # DDPM
    "2112.10752",  # Latent diffusion
    "2302.13971",  # LLaMA
    "2201.11903",  # Chain-of-thought prompting
    "1806.07366",  # Neural ODEs
    "1312.5602",  # DQN
    "1711.05101",  # AdamW
    "1505.04597",  # U-Net
    # mathematics and physics
    "math/0211159",  # Perelman, entropy formula for Ricci flow
    "math/0303109",  # Perelman, Ricci flow with surgery
    "math/0404188",  # Green-Tao, primes contain arbitrarily long APs
    "1311.4600",  # Maynard, small gaps between primes
    "1303.5113",  # Hairer, a theory of regularity structures
    "1306.3969",  # Marcus-Spielman-Srivastava, Kadison-Singer
    "1603.04246",  # Viazovska, sphere packing in dimension 8
    "1312.7748",  # Helfgott, ternary Goldbach
    "1602.03837",  # LIGO, observation of gravitational waves
    "hep-th/9711200",  # Maldacena, large N limit
)


@dataclass
class Paper:
    arxiv_id: str
    group: str
    cohort: str | None = None  # "2022" / "2024" for benchmark and same_vintage
    title: str = ""
    abstract: str = ""
    primary_category: str = ""
    published: str = ""
    problem_ids: list[str] = field(default_factory=list)
    theorem_field: str | None = None  # RealMath ``theorem`` text used to locate the statement
    latex_available: bool = False
    theorem_env: str | None = None
    theorem_statement: str | None = None
    theorem_matched_by: str | None = None
    pretext: str | None = None
    proof_start: str | None = None


# --------------------------------------------------------------------------- arXiv access


class ArxivClient:
    """Minimal, rate-limited arXiv API + e-print client with an on-disk cache."""

    def __init__(self, cache_dir: Path, *, delay_sec: float = ARXIV_DELAY_SEC, log=print) -> None:
        self.cache_dir = cache_dir
        self.delay_sec = delay_sec
        self.log = log
        self._last_request = 0.0

    @staticmethod
    def safe_id(arxiv_id: str) -> str:
        return arxiv_id.replace("/", "_")

    def _paper_dir(self, arxiv_id: str) -> Path:
        path = self.cache_dir / self.safe_id(arxiv_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get(self, url: str) -> bytes:
        wait = self.delay_sec - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        # arXiv answers 406/429 intermittently once it decides a client is too eager;
        # both clear after a pause, so back off generously instead of giving up.
        attempts = 6
        for attempt in range(attempts):
            try:
                payload = _http_get(url)
                self._last_request = time.monotonic()
                return payload
            except FileNotFoundError:
                self._last_request = time.monotonic()
                raise
            except Exception as exc:  # noqa: BLE001 - retry with backoff, then re-raise
                self._last_request = time.monotonic()
                if attempt == attempts - 1:
                    raise
                pause = 20 * (attempt + 1)
                self.log(f"  retry {attempt + 1} in {pause}s for {url[:120]}: {str(exc)[:80]}")
                time.sleep(pause)
        raise RuntimeError("unreachable")

    def metadata(self, arxiv_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Fetch title/abstract/category for ids (cached per paper as meta.json)."""

        result: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for arxiv_id in arxiv_ids:
            meta_path = self._paper_dir(arxiv_id) / "meta.json"
            if meta_path.is_file():
                result[arxiv_id] = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                missing.append(arxiv_id)
        for chunk_start in range(0, len(missing), 50):
            chunk = missing[chunk_start : chunk_start + 50]
            self.log(f"  arXiv metadata for {len(chunk)} papers ...")
            url = f"{ARXIV_API}?id_list={','.join(chunk)}&max_results={len(chunk)}"
            for entry in parse_atom(self._get(url)):
                result[entry["arxiv_id"]] = entry
                (self._paper_dir(entry["arxiv_id"]) / "meta.json").write_text(
                    json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        return result

    def search(self, query: str, *, max_results: int, sort_by: str = "submittedDate") -> list[dict[str, Any]]:
        params = urllib.parse.urlencode(
            {"search_query": query, "start": 0, "max_results": max_results, "sortBy": sort_by, "sortOrder": "descending"}
        )
        return parse_atom(self._get(f"{ARXIV_API}?{params}"))

    def eprint_source(self, arxiv_id: str) -> str | None:
        """Return the LaTeX source text (cached as source.tex); None when PDF-only."""

        paper_dir = self._paper_dir(arxiv_id)
        source_path = paper_dir / "source.tex"
        marker_path = paper_dir / "no_source"
        if source_path.is_file():
            return source_path.read_text(encoding="utf-8")
        if marker_path.is_file():
            return None
        self.log(f"  e-print {arxiv_id} ...")
        try:
            blob = self._get(ARXIV_EPRINT + arxiv_id)
        except FileNotFoundError:
            marker_path.write_text("404 from arxiv.org/e-print\n", encoding="utf-8")
            return None
        text = extract_tex_sources(blob)
        if text is None:
            marker_path.write_text("PDF-only or unreadable e-print\n", encoding="utf-8")
            return None
        source_path.write_text(text, encoding="utf-8")
        return text


def _http_get(url: str) -> bytes:
    """GET through the system ``curl``.

    Python's TLS stack gets intermittent 406/429 answers from export.arxiv.org even
    with byte-identical headers, while curl is served every time, so we shell out.
    """

    result = subprocess.run(
        ["curl", "-sS", "-L", "--fail-with-body", "--max-time", "120", "-A", USER_AGENT, url],
        capture_output=True,
        check=False,
    )
    if result.returncode == 22 and "404" in result.stderr.decode(errors="replace"):
        raise FileNotFoundError(url)
    if result.returncode != 0:
        raise RuntimeError(f"curl exit {result.returncode} for {url[:120]}: {result.stderr.decode(errors='replace')[:120]}")
    return result.stdout


def parse_atom(payload: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    entries: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", ATOM):
        raw_id = (entry.findtext("a:id", default="", namespaces=ATOM) or "").strip()
        arxiv_id = re.sub(r"^.*/abs/", "", raw_id)
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        if not arxiv_id:
            continue
        primary = entry.find("arxiv:primary_category", ATOM)
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": " ".join((entry.findtext("a:title", default="", namespaces=ATOM) or "").split()),
                "abstract": " ".join((entry.findtext("a:summary", default="", namespaces=ATOM) or "").split()),
                "primary_category": primary.get("term", "") if primary is not None else "",
                "published": (entry.findtext("a:published", default="", namespaces=ATOM) or "")[:10],
                "authors": [
                    " ".join((author.findtext("a:name", default="", namespaces=ATOM) or "").split())
                    for author in entry.findall("a:author", ATOM)
                ],
            }
        )
    return entries


# --------------------------------------------------------------------------- group construction


def cohort_of(arxiv_id: str) -> str | None:
    return {"2205": "2022", "2206": "2022", "2412": "2024"}.get(arxiv_id[:4])


def load_benchmark_papers(problems_path: Path, realmath_path: Path) -> list[Paper]:
    problems = json.loads(problems_path.read_text(encoding="utf-8"))
    realmath = json.loads(realmath_path.read_text(encoding="utf-8"))
    rows = realmath if isinstance(realmath, list) else realmath["problems"]
    by_id = {row["id"]: row for row in rows}
    papers: dict[str, Paper] = {}
    for problem in problems:
        row = by_id[problem["id"]]
        arxiv_id = str(row["arxiv_id"])
        paper = papers.setdefault(
            arxiv_id,
            Paper(arxiv_id=arxiv_id, group="benchmark", cohort=cohort_of(arxiv_id), theorem_field=row.get("theorem")),
        )
        paper.problem_ids.append(problem["id"])
    return list(papers.values())


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of ``total`` slots proportional to ``counts``."""

    grand = sum(counts.values())
    if grand == 0 or total <= 0:
        return {key: 0 for key in counts}
    raw = {key: total * value / grand for key, value in counts.items()}
    result = {key: int(value) for key, value in raw.items()}
    for key in sorted(counts, key=lambda k: raw[k] - result[k], reverse=True)[: total - sum(result.values())]:
        result[key] += 1
    return result


def _date_query(start: date, end: date) -> str:
    return f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"


def sample_controls(
    client: ArxivClient,
    *,
    group: str,
    category_counts: dict[str, int],
    total: int,
    window: tuple[date, date],
    exclude: set[str],
    rng: random.Random,
    cohort: str | None,
    pool_size: int = 150,
) -> list[Paper]:
    """Sample ``total`` papers proportionally to ``category_counts`` inside a date window."""

    papers: list[Paper] = []
    for category, want in allocate(category_counts, total).items():
        if want == 0:
            continue
        query = f"cat:{category} AND {_date_query(*window)}"
        client.log(f"  {group}: {category} x{want} from {window[0]}..{window[1]}")
        pool = [
            entry
            for entry in client.search(query, max_results=pool_size)
            if entry["arxiv_id"] not in exclude and entry["primary_category"] == category
        ]
        # Keep only papers first submitted inside the window (the API also matches later versions).
        pool = [entry for entry in pool if window[0].isoformat() <= entry["published"] <= window[1].isoformat()] or pool
        rng.shuffle(pool)
        for entry in pool[:want]:
            papers.append(
                Paper(
                    arxiv_id=entry["arxiv_id"],
                    group=group,
                    cohort=cohort,
                    title=entry["title"],
                    abstract=entry["abstract"],
                    primary_category=entry["primary_category"],
                    published=entry["published"],
                )
            )
            (client._paper_dir(entry["arxiv_id"]) / "meta.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return papers


def fill_metadata(client: ArxivClient, papers: Iterable[Paper]) -> None:
    papers = list(papers)
    meta = client.metadata([paper.arxiv_id for paper in papers if not paper.abstract])
    for paper in papers:
        entry = meta.get(paper.arxiv_id)
        if entry is None:
            continue
        paper.title = entry["title"]
        paper.abstract = entry["abstract"]
        paper.primary_category = entry["primary_category"]
        paper.published = entry["published"]


def fill_sources(client: ArxivClient, papers: Iterable[Paper]) -> None:
    for paper in papers:
        try:
            source = client.eprint_source(paper.arxiv_id)
        except Exception as exc:  # noqa: BLE001 - a missing source only disables two probes
            client.log(f"  e-print failed for {paper.arxiv_id}: {exc}")
            source = None
        if source is None:
            continue
        paper.latex_available = True
        context = theorem_context(source, paper.theorem_field)
        if context is None:
            continue
        paper.theorem_env = context.env_name
        paper.theorem_statement = context.statement
        paper.theorem_matched_by = context.matched_by
        paper.pretext = context.pretext
        paper.proof_start = context.proof_start


# --------------------------------------------------------------------------- CLI


def reparse_corpus(args: argparse.Namespace) -> list[Paper]:
    """Re-derive the LaTeX fields of an existing corpus.json from the cache (no network)."""

    client = ArxivClient(args.output_dir / "cache", delay_sec=args.delay)
    payload = json.loads((args.output_dir / "corpus.json").read_text(encoding="utf-8"))
    papers: list[Paper] = []
    for row in payload["papers"]:
        paper = Paper(**{key: row.get(key) for key in Paper.__dataclass_fields__ if key in row})
        paper.latex_available = False
        paper.theorem_env = paper.theorem_statement = paper.theorem_matched_by = paper.pretext = paper.proof_start = None
        papers.append(paper)
    fill_sources(client, papers)
    return papers


def update_famous(args: argparse.Namespace) -> list[Paper]:
    """Append famous papers missing from an existing corpus.json (fetches only the new ones)."""

    client = ArxivClient(args.output_dir / "cache", delay_sec=args.delay)
    payload = json.loads((args.output_dir / "corpus.json").read_text(encoding="utf-8"))
    papers = [Paper(**{key: row.get(key) for key in Paper.__dataclass_fields__ if key in row}) for row in payload["papers"]]
    have = {paper.arxiv_id for paper in papers}
    new = [Paper(arxiv_id=arxiv_id, group="famous") for arxiv_id in FAMOUS_ARXIV_IDS if arxiv_id not in have]
    print(f"famous: {len(new)} new papers")
    fill_metadata(client, new)
    if not args.no_eprints:
        fill_sources(client, new)
    return papers + new


def build_corpus(args: argparse.Namespace) -> list[Paper]:
    if args.reparse:
        return reparse_corpus(args)
    if args.update_famous:
        return update_famous(args)
    client = ArxivClient(args.output_dir / "cache", delay_sec=args.delay)
    rng = random.Random(args.seed)
    papers: list[Paper] = []

    benchmark = load_benchmark_papers(args.problems, args.realmath)
    print(f"benchmark: {len(benchmark)} papers, {sum(len(p.problem_ids) for p in benchmark)} problems")
    fill_metadata(client, benchmark)
    exclude = {paper.arxiv_id for paper in benchmark}
    # Exclude every RealMath paper, not only the 88 we evaluate.
    realmath = json.loads(args.realmath.read_text(encoding="utf-8"))
    exclude |= {str(row["arxiv_id"]) for row in (realmath if isinstance(realmath, list) else realmath["problems"])}
    if "benchmark" in args.groups:
        papers.extend(benchmark)

    category_counts_by_cohort: dict[str, dict[str, int]] = {}
    for paper in benchmark:
        if paper.cohort and paper.primary_category:
            counts = category_counts_by_cohort.setdefault(paper.cohort, {})
            counts[paper.primary_category] = counts.get(paper.primary_category, 0) + 1
    all_counts: dict[str, int] = {}
    for counts in category_counts_by_cohort.values():
        for category, value in counts.items():
            all_counts[category] = all_counts.get(category, 0) + value

    if "same_vintage" in args.groups:
        per_cohort = allocate({c: len([p for p in benchmark if p.cohort == c]) for c in COHORT_WINDOWS}, args.n_same_vintage)
        for cohort, window in COHORT_WINDOWS.items():
            papers.extend(
                sample_controls(
                    client,
                    group="same_vintage",
                    category_counts=category_counts_by_cohort.get(cohort, all_counts),
                    total=per_cohort[cohort],
                    window=window,
                    exclude=exclude,
                    rng=rng,
                    cohort=cohort,
                )
            )
    if "fresh" in args.groups:
        end = date.today()
        papers.extend(
            sample_controls(
                client,
                group="fresh",
                category_counts=all_counts,
                total=args.n_fresh,
                window=(end - timedelta(days=args.fresh_window_days), end),
                exclude=exclude,
                rng=rng,
                cohort=None,
            )
        )
    if "famous" in args.groups:
        famous = [Paper(arxiv_id=arxiv_id, group="famous") for arxiv_id in FAMOUS_ARXIV_IDS]
        fill_metadata(client, famous)
        papers.extend(famous)

    if not args.no_eprints:
        fill_sources(client, papers)
    return papers


def summarize(papers: Sequence[Paper]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for group in GROUPS:
        members = [paper for paper in papers if paper.group == group]
        if not members:
            continue
        summary[group] = {
            "papers": len(members),
            "with_abstract": sum(1 for p in members if p.abstract),
            "with_latex": sum(1 for p in members if p.latex_available),
            "with_theorem": sum(1 for p in members if p.theorem_statement),
            "with_pretext": sum(1 for p in members if p.pretext),
            "with_proof": sum(1 for p in members if p.proof_start),
            "theorem_matched_by_field": sum(1 for p in members if p.theorem_matched_by == "field"),
        }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS, help="The 133 evaluated problems.")
    parser.add_argument("--realmath", type=Path, default=DEFAULT_REALMATH, help="Full RealMath rows (arxiv_id, theorem).")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--n-same-vintage", type=int, default=40)  # same size as the famous group
    parser.add_argument("--n-fresh", type=int, default=40)
    parser.add_argument("--fresh-window-days", type=int, default=28, help="Fresh papers are at most this old.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=ARXIV_DELAY_SEC, help="Seconds between arXiv requests.")
    parser.add_argument("--no-eprints", action="store_true", help="Skip LaTeX sources (abstract probes only).")
    parser.add_argument("--reparse", action="store_true", help="Recompute theorem/pretext/proof fields from the cached sources only.")
    parser.add_argument("--update-famous", action="store_true", help="Add famous papers missing from corpus.json; everything else is kept.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    papers = build_corpus(args)
    output_path = args.output_dir / "corpus.json"
    payload = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "args": {key: str(value) for key, value in vars(args).items()},
        "summary": summarize(papers),
        "papers": [asdict(paper) for paper in papers],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
