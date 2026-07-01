# /// script
# requires-python = ">=3.12"
# dependencies = ["zerospeech-libriabx2==0.9.8"]
# ///
import argparse
import sys
import time
import tempfile
from datetime import timedelta

from zrc_abx2 import EvalArgs, EvalABX

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("item")
    parser.add_argument("root")
    args = parser.parse_args()

    tmpdir = tempfile.TemporaryDirectory()
    start = time.perf_counter()
    score = EvalABX().eval_abx(
        EvalArgs(
            path_data=args.root,
            path_item_file=args.item,
            path_checkpoint=None,
            file_extension=".pt",
            feature_size=0.02,
            cuda=True,
            speaker_mode="within",
            context_mode="within",
            distance_mode="cosine",
            max_size_group=sys.maxsize,
            max_x_across=sys.maxsize,
            out=tmpdir.name,
            seed=0,
            pooling="none",
        )
    )[0]["score"]
    end = time.perf_counter()
    tmpdir.cleanup()
    print(f"Score: {score:.3%}")
    print(f"Time: {timedelta(seconds=end - start)}")
