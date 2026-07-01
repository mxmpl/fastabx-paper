import argparse
import multiprocessing
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from ABXpy.analyze import analyze
from ABXpy.score import score
from ABXpy.task import Task
from ABXpy.distance import run as distance


def collapse_score(path):
    df = pd.read_csv(path, sep="\t").rename(columns={"phone_1": "#phone", "phone_2": "#phone_b", "n": "size"})
    df["score"] = 1 - df["score"]
    parts = df["by"].str.strip("()").str.split(", ", n=2, expand=True)
    df["prev-phone"] = parts[0].str.strip("'")
    df["next-phone"] = parts[1].str.strip("'")
    df["speaker"] = parts[2].astype("int64")
    df = df[["#phone", "speaker", "next-phone", "prev-phone", "#phone_b", "score", "size"]]
    for level in [("next-phone", "prev-phone"), ("speaker",)]:
        group_key = [c for c in df.columns if c not in ("score", "size", *level)]
        df = df.groupby(group_key, sort=False, as_index=False).agg({"score": "mean", "size": "sum"})
    return df["score"].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("item")
    parser.add_argument("root")
    parser.add_argument("--njobs", type=int, default=4)
    args = parser.parse_args()
    if args.njobs > 1:  # HDF5 is not fork-safe
        multiprocessing.set_start_method("forkserver")

    item, root = Path(args.item), Path(args.root)
    path_task, path_features = str(root / "task.abx"), str(root / "hubert.features")
    path_dist, path_score = str(root / "hubert.distance"), str(root / "triplets.score")
    path_csv = str(root / "results.csv")

    start = time.perf_counter()
    print("TASK")
    task = Task(args.item, "phone", by=["prev-phone", "next-phone", "speaker"], across=None, verbose=True)
    task.generate_triplets(path_task)
    print("DISTANCE")
    distance(path_features, path_task, path_dist, normalized=1, njobs=args.njobs)
    print("SCORE")
    score(path_task, path_dist, path_score)
    print("ANALYZE")
    analyze(path_task, path_score, path_csv)
    end = time.perf_counter()

    score = collapse_score(path_csv)
    print(f"Score: {score:.3%}")
    print(f"Time: {timedelta(seconds=end - start)}")
