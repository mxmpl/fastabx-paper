# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastabx>=0.4.1",
#     "matplotlib>=3.10.1",
#     "numpy>=2.2.4",
# ]
# ///
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastabx import Dataset, Score, Task

if __name__ == "__main__":
    assets = Path("./assets")
    plt.style.use(assets / "paper.mplstyle")
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.top"] = True

    n, mu, cov = 50, np.zeros(2), np.array([[1, 0], [0, 1]])
    rng = np.random.default_rng(seed=1)
    first = rng.multivariate_normal(mu, cov, n)
    second = rng.multivariate_normal(mu, cov, n)

    fig, ax = plt.subplots(figsize=(455.24 / 72.27, 1.25), ncols=5, sharex=True, sharey=True)
    fig.set_layout_engine("tight", pad=0.1, w_pad=1.5)
    for k, shift in enumerate([0, 1, 2, 3, 4]):
        second_shifted = second + np.full(2, shift)
        dataset = Dataset.from_numpy(np.vstack([first, second_shifted]), {"label": [0] * n + [1] * n})
        task = Task(dataset, on="label")
        score = Score(task, "euclidean")
        ax[k].scatter(*first.T, facecolors="none", edgecolors="C0", s=10, lw=0.7)
        ax[k].scatter(*second_shifted.T, facecolors="none", edgecolors="C1", s=10, lw=0.7)
        title = ax[k].set_title(rf"$\mu={shift}$: ${score.collapse() * 100:.2f}\%$", fontsize=10)
        ax[k].set_xlim(-3, 5)
        ax[k].set_ylim(-3, 5)
        ax[k].set_xticks([-2, 0, 2, 4])
        ax[k].set_yticks([-2, 0, 2, 4])
        ax[k].xaxis.set_tick_params(labelsize=9)
        ax[k].yaxis.set_tick_params(labelsize=9)
    plt.savefig("gaussians.pdf")
    plt.close()
