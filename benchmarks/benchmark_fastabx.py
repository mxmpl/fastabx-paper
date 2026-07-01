# /// script
# requires-python = ">=3.12"
# dependencies = ["fastabx==0.8.0"]
# ///
import argparse
import time
from datetime import timedelta

import torch
from fastabx import zerospeech_abx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("item")
    parser.add_argument("root")
    args = parser.parse_args()

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
    print(f"Score: {score:.3%}")
    print(f"Time: {timedelta(seconds=end - start)}")
