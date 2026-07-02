# /// script
# requires-python = ">=3.12"
# dependencies = ["zerospeech-libriabx2==0.9.8", "psutil"]
# ///
import argparse
import sys
import tempfile
from pathlib import Path

import torch
from benchmark_utils import run_benchmark, subset_from_item, write_result
from zrc_abx2 import EvalArgs, EvalABX

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("item")
    parser.add_argument("root")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--snapshot-dir", default=None)
    args = parser.parse_args()

    subset = subset_from_item(args.item)
    snapshot_dir = args.snapshot_dir or Path(args.output).parent / "snapshots"
    with_cuda = torch.cuda.is_available()

    def run():
        with tempfile.TemporaryDirectory() as tmpdir:
            return EvalABX().eval_abx(
                EvalArgs(
                    path_data=args.root,
                    path_item_file=args.item,
                    path_checkpoint=None,
                    file_extension=".pt",
                    feature_size=0.02,
                    cuda=with_cuda,
                    speaker_mode="within",
                    context_mode="within",
                    distance_mode="cosine",
                    max_size_group=sys.maxsize,
                    max_x_across=sys.maxsize,
                    out=tmpdir,
                    seed=0,
                    pooling="none",
                )
            )[0]["score"]

    score, stats = run_benchmark(run, with_cuda=with_cuda, runs=args.runs)
    write_result(args.output, "librilight", args.item, score, stats)
