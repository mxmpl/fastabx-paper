# Companion repository to fastabx paper

- Library: https://github.com/bootphon/fastabx
- Documentation: https://docs.cognitive-ml.fr/fastabx
- Paper: https://arxiv.org/abs/2505.02692

## Benchmark fastabx against ABXpy and Libri-Light ABX

The benchmarks are fully automated with [pixi](https://pixi.sh). From the
`benchmarks/` directory, run the whole pipeline on a LibriSpeech subset
(`dev-clean` or `dev-other`):

```bash
cd benchmarks
pixi run benchmark dev-clean
```

Each pinned dependency lives in its own environment, so a single command:

1. downloads and extracts the LibriSpeech subset,
2. extracts the HuBERT features (`.pt` files and an h5features archive),
3. runs the three benchmarks: fastabx (with and without the Libri-Light bug), ABXpy, and Libri-Light ABX.

Individual steps are available as separate tasks (`pixi task list`).

## Reproduce figures from the paper

This saves figures to PDF files:

```bash
uv run figures/gaussians.py
uv run figures/correlation.py
uv run figures/phoneme_vs_speaker.py
```

## Citation

```bibtex
@misc{fastabx,
  title={fastabx: A library for efficient computation of ABX discriminability},
  author={Maxime Poli and Emmanuel Chemla and Emmanuel Dupoux},
  year={2025},
  eprint={2505.02692},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.02692},
}
```
