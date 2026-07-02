# /// script
# requires-python = ">=3.12"
# dependencies = ["fastabx==0.8.0", "psutil"]
# ///
import argparse
import os
from pathlib import Path

import torch
from benchmark_utils import run_benchmark, subset_from_item, write_result
from fastabx import zerospeech_abx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("item")
    parser.add_argument("root")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--snapshot-dir", default=None)
    args = parser.parse_args()

    benchmark = "fastabx-librilight-bug" if os.environ.get("FASTABX_WITH_LIBRILIGHT_BUG") == "1" else "fastabx"
    subset = subset_from_item(args.item)
    snapshot_dir = args.snapshot_dir or Path(args.output).parent / "snapshots"

    def run():
        return zerospeech_abx(
            args.item,
            args.root,
            max_size_group=None,
            max_x_across=None,
            speaker="within",
            context="within",
            distance="angular",
            frequency=50,
            feature_maker=torch.load,
            extension=".pt",
            seed=0,
        )

    score, stats = run_benchmark(run, with_cuda=torch.cuda.is_available(), runs=args.runs)
    write_result(args.output, benchmark, args.item, score, stats)
