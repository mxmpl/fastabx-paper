import argparse
import time
import polars as pl
import polars.selectors as cs
from datetime import timedelta
from pathlib import Path

from ABXpy.analyze import analyze
from ABXpy.score import score
from ABXpy.task import Task
from ABXpy.distance import run as distance


def collapse_score(path):
    extract = (
        pl.col("by")
        .str.strip_chars("()")
        .str.split_exact(", ", 2)
        .struct.rename_fields(["prev-phone", "next-phone", "speaker"])
        .struct.unnest()
    )
    df = (
        pl.scan_csv(path, separator="\t")
        .rename({"phone_1": "#phone", "phone_2": "#phone_b", "n": "size"})
        .with_columns((1 - pl.col("score")).alias("score"), extract)
        .with_columns(
            pl.col("prev-phone").str.strip_chars("'"),
            pl.col("next-phone").str.strip_chars("'"),
            pl.col("speaker").cast(pl.Int64),
        )
        .select("#phone", "speaker", "next-phone", "prev-phone", "#phone_b", "score", "size")
        .sort("#phone", "#phone_b", "speaker", "next-phone", "prev-phone")
        .collect()
    )
    for level in [("next-phone", "prev-phone"), ("speaker",)]:
        group_key = cs.exclude("score", "size", *level)
        df = df.group_by(group_key, maintain_order=True).agg(pl.col("score").mean(), pl.col("size").sum())
    return df["score"].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("item")
    parser.add_argument("root")
    args = parser.parse_args()

    item, root = Path(args.item), Path(args.root)
    path_task, path_features = str(root / "task.abx"), str(root / "hubert.features")
    path_dist, path_score = str(root / "hubert.distance"), str(root / "triplets.score")
    path_csv = str(root / "results.csv")

    start = time.perf_counter()
    print("TASK")
    task = Task(args.item, "phone", by=["prev-phone", "next-phone", "speaker"], across=None, verbose=True)
    task.generate_triplets(path_task)
    print("DISTANCE")
    distance(path_features, path_task, path_dist, normalized=1, njobs=1)  # njobs > 1 crashes
    print("SCORE")
    score(path_task, path_dist, path_score)
    print("ANALYZE")
    analyze(path_task, path_score, path_csv)
    end = time.perf_counter()

    score = collapse_score(path_csv)
    print(f"Score: {score:.3%}")
    print(f"Time: {timedelta(seconds=end - start)}")
