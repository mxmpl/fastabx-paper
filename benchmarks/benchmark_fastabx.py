# /// script
# requires-python = ">=3.12"
# dependencies = ["fastabx==0.8.0"]
# ///
import argparse
import os
import time

import torch
from benchmark_utils import write_result
from fastabx import zerospeech_abx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("item")
    parser.add_argument("root")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    benchmark = "fastabx-librilight-bug" if os.environ.get("FASTABX_WITH_LIBRILIGHT_BUG") == "1" else "fastabx"

    start = time.perf_counter()
    score = zerospeech_abx(
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
    end = time.perf_counter()
    write_result(args.output, benchmark, args.item, score, end - start)
