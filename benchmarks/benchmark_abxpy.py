import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
from ABXpy.analyze import analyze
from ABXpy.distance import run as distance
from ABXpy.score import score
from ABXpy.task import Task
from benchmark_utils import run_benchmark, write_result


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
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if args.njobs > 1:  # HDF5 is not fork-safe
        multiprocessing.set_start_method("forkserver")

    root = Path(args.root)
    path_task, path_features = str(root / "task.abx"), str(root / "hubert.features")
    path_dist, path_score = str(root / "hubert.distance"), str(root / "triplets.score")
    path_csv = str(root / "results.csv")

    def run():
        for stale in (path_task, path_dist, path_score, path_csv):
            Path(stale).unlink(missing_ok=True)
        print("TASK")
        task = Task(args.item, "phone", by=["prev-phone", "next-phone", "speaker"], across=None, verbose=True)
        task.generate_triplets(path_task)
        print("DISTANCE")
        distance(path_features, path_task, path_dist, normalized=1, njobs=args.njobs)
        print("SCORE")
        score(path_task, path_dist, path_score)
        print("ANALYZE")
        analyze(path_task, path_score, path_csv)
        return collapse_score(path_csv)

    result, stats = run_benchmark(run, with_cuda=False, runs=args.runs)
    write_result(args.output, "abxpy", args.item, result, stats)
