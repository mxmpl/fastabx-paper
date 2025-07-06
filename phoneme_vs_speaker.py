# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.1",
#     "numpy>=2.2.4",
#     "polars>=1.27.1",
# ]
# ///
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs

if __name__ == "__main__":
    assets = Path("./assets")
    plt.style.use(assets / "paper.mplstyle")
    plt.rcParams.update({"lines.linewidth": 1, "lines.markersize": 3.5})

    trajectories = (
        pl.read_csv(assets / "phoneme_vs_speaker.csv")
        .group_by(cs.exclude("dataset", "speaker", "triphone"), maintain_order=True)
        .agg(pl.col("speaker").mean(), pl.col("triphone").mean())
        .with_columns(pl.col("speaker") * 100, pl.col("triphone") * 100)
    )

    def filter_traj(df: pl.DataFrame, model: str) -> np.ndarray:
        return df.filter(pl.col("model") == model).sort(by="layer").select("speaker", "triphone").to_numpy()

    fig, ax = plt.subplots(figsize=(455.24 / 72.27 * 0.48, 2.3))
    fig.set_layout_engine("tight", pad=0)
    hubert = filter_traj(trajectories, "hubert-base")
    ax.plot(*hubert.T, marker="o", label="HuBERT")
    spinhubert = np.vstack([hubert[10][np.newaxis, :], filter_traj(trajectories, "spinhubert-2048")])
    ax.plot(*spinhubert.T, marker="o", color="C0", linestyle="--", label=r"HuBERT + Spin$_{2048}$")
    wavlm = filter_traj(trajectories, "wavlm-base")
    ax.plot(*wavlm.T, marker="o", label="WavLM")
    spinwavlm = np.vstack([wavlm[10][np.newaxis, :], filter_traj(trajectories, "spinwavlm-2048")])
    ax.plot(*spinwavlm.T, marker="o", color="C1", linestyle="--", label=r"WavLM + Spin$_{2048}$")
    ax.scatter(*filter_traj(trajectories, "mfcc").T, marker="x", color="black", label="MFCC")
    ax.set_ylim(2, 14)
    ax.set_xlabel(r"ABX on speaker (\%)", fontsize=10)
    ax.set_ylabel(r"ABX on phoneme (\%)", fontsize=10)
    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    ax.legend(fontsize=9)
    plt.savefig("trajectory.pdf")
    plt.close()
