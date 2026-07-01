"""Shared helper to collect benchmark results into a single JSON file.

Imported by all three benchmark scripts, which each run in their own pixi
environment but share this directory.
"""

import json
from pathlib import Path


def subset_from_item(item):
    """Derive the LibriSpeech subset name from an item path (triphone-<subset>.item)."""
    stem = Path(item).stem
    prefix = "triphone-"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def write_result(output, benchmark, item, score, elapsed):
    """Upsert one result into the JSON list at ``output``.

    Results are keyed by (benchmark, subset) so re-running a benchmark replaces
    its previous entry instead of appending a duplicate.
    """
    output = Path(output)
    subset = subset_from_item(item)
    record = {
        "benchmark": benchmark,
        "subset": subset,
        "score": score,
        "time_seconds": elapsed,
    }
    results = []
    if output.exists():
        results = json.loads(output.read_text())
    results = [r for r in results if (r["benchmark"], r["subset"]) != (benchmark, subset)]
    results.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
