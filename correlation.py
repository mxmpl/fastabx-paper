# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastabx>=0.4.1",
#     "matplotlib>=3.10.1",
#     "numpy>=2.2.4",
#     "scipy>=1.15.2",
#     "torch>=2.7.1",
# ]
# ///
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from fastabx import Dataset, Score, Task
from scipy.stats import linregress, pearsonr

type Feature = Literal["-", "0", "+"]
type ArticulatoryFeatures = dict[str, dict[str, Feature]]


def read_articulatory_features(path: str | Path) -> ArticulatoryFeatures:
    articulatory_features = pl.read_csv(path, infer_schema_length=0)
    return dict(
        zip(
            articulatory_features["ipa"].to_list(),
            articulatory_features.select(pl.exclude("ipa")).to_dicts(),
            strict=True,
        )
    )


def feature_distance(features: ArticulatoryFeatures, x: str, y: str) -> np.float64 | None:
    if x not in features or y not in features:
        return None
    xt = np.asarray([{"-": -1, "0": 0, "+": 1}[i] for i in features[x].values()])
    yt = np.asarray([{"-": -1, "0": 0, "+": 1}[i] for i in features[y].values()])
    return np.abs(xt - yt).mean() / 2


def build_scores_and_distances(
    score: Score,
    features: ArticulatoryFeatures,
    arpabet_to_ipa: pl.DataFrame,
) -> pl.DataFrame:
    return (
        score.details(levels=[("prev-phone", "next-phone"), "speaker"])
        .lazy()
        .with_columns(
            pl.when(pl.col("#phone") < pl.col("#phone_b"))
            .then(pl.struct(pl.col("#phone").alias("first"), pl.col("#phone_b").alias("second")))
            .otherwise(pl.struct(pl.col("#phone_b").alias("first"), pl.col("#phone").alias("second")))
            .alias("pair")
        )
        .group_by("pair", maintain_order=True)
        .agg(score=pl.col("score").mean(), size=pl.col("size").sum(), full_pair=pl.len() == 2)
        .with_columns(pl.col("pair").struct.unnest())
        .select(pl.exclude("pair"))
        .with_columns(
            pl.col("first").replace_strict(arpabet_to_ipa["arpabet"], arpabet_to_ipa["ipa"]),
            pl.col("second").replace_strict(arpabet_to_ipa["arpabet"], arpabet_to_ipa["ipa"]),
            pl.col("score") * 100,
        )
        .with_columns(
            pl.struct("first", "second")
            .map_elements(lambda x: feature_distance(features, x["first"], x["second"]), return_dtype=pl.Float64)
            .alias("distance")
        )
        .drop_nulls()
        .collect()
    )


if __name__ == "__main__":
    assets = Path("./assets")

    features = read_articulatory_features(assets / "articulatory_features.csv")
    arpabet_to_ipa = pl.read_csv(assets / "arpabet_to_ipa.csv")

    dataset = Dataset.from_item(assets / "triphone-dev-clean.item", assets / "hubert", 50)
    task = Task(dataset, on="#phone", by=["speaker", "next-phone", "prev-phone"])
    score = Score(task, "angular")

    print(f"No features available for: {set(arpabet_to_ipa['ipa']) - set(features)}")
    df = build_scores_and_distances(score, features, arpabet_to_ipa)
    coeff = pearsonr(df["distance"], df["score"])[0]
    line = linregress(df["distance"], df["score"])
    x = np.array([0, 0.5])

    plt.style.use(assets / "paper.mplstyle")
    fig, ax = plt.subplots(figsize=(455.24 / 72.27 * 0.48, 1.7))
    fig.set_layout_engine("tight", pad=0)
    ax.scatter(df["distance"], df["score"], alpha=0.5, s=15, edgecolors="none", rasterized=True)
    ax.plot(x, line.intercept + line.slope * x, "C1", linewidth=1.5)
    ax.text(0.39, 11, rf"\boldmath $ r ={coeff:.3f}$", color="C1")
    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    ax.set_ylabel(r"ABX error rate (\%)")
    ax.set_xlabel("Distance between articulatory features")
    plt.savefig("correlation.pdf", dpi=400)
    plt.close()
