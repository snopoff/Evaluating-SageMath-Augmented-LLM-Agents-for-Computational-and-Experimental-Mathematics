"""Reproduce category retention for the released normalized pool and evaluation set."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis/generated')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / 'data/construction/source_metadata.json').read_text())
    tags = json.loads((ROOT / 'data/construction/arxiv_categories_633.json').read_text())
    categories = {row['id']: tags[row['arxiv_id']]['primary'] for row in source}
    pool = json.loads((ROOT / 'data/processed/normalized_realmath.json').read_text())
    evaluation = json.loads((ROOT / 'data/processed/problems_to_run.json').read_text())
    stages = {
        'source_633': [row['id'] for row in source],
        'normalized_219': [row['id'] for row in pool],
        'evaluation_133': [row['id'] for row in evaluation],
    }
    assert len(set(stages['normalized_219'])) == 219
    assert len(set(stages['evaluation_133'])) == 133
    assert set(stages['evaluation_133']) <= set(stages['normalized_219'])
    counts = {name: Counter(categories[item] for item in ids) for name, ids in stages.items()}
    rows = [
        {'category': category, **{name: counts[name][category] for name in stages}}
        for category in sorted(set(categories.values()))
    ]
    with (args.output_dir / 'category_retention.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        'counts': {name: len(ids) for name, ids in stages.items()},
        'evaluation_subset_of_normalized': True,
        'analysis_scope': 'Category counts for the released source metadata, normalized pool and exact evaluation set; retention 219 to 133.',
    }
    (args.output_dir / 'construction_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
