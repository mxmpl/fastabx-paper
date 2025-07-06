# Companion repository to fastabx paper

- Library: https://github.com/bootphon/fastabx
- Documentation: https://docs.cognitive-ml.fr/fastabx
- Paper: https://arxiv.org/abs/2505.02692

## Benchmark fastabx against ABXpy and Libri-Light ABX

### Setup

You need to have downloaded [LibriSpeech dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz).

Extract HuBERT features (replace the path to LibriSpeech dev-clean with your actual path) and format them to h5features:
```bash
uv run extract_features.py /path/to/LibriSpeech/dev-clean ./assets/hubert
uv run https://raw.githubusercontent.com/bootphon/fastabx/refs/heads/main/scripts/convert_features.py h5 ./assets/hubert ./assets/abxpy/hubert.h5f --step 0.02
```

### fastabx vs ABXpy

To benchmark ABXpy, create a conda environment with the exact dependencies and run the full pipeline (~2h):
```bash
micromamba create -f ./assets/abxpy.yml
micromamba activate ABXpy
python run_abxpy.py ./assets/triphone-dev-clean.item ./assets/abxpy
micromamba deactivate
```

And fastabx (~2min):
```bash
uv run benchmark_fastabx.py ./assets/triphone-dev-clean.item ./assets/hubert
```

### fastabx vs Libri-Light ABX

To benchmark Libri-Light (~5min):

```bash
CC="gcc" uv run benchmark_librilight.py ./assets/triphone-dev-clean.item ./assets/hubert
```

And fastabx (~2min):
```bash
FASTABX_WITH_LIBRILIGHT_BUG=1 uv run benchmark_fastabx.py ./assets/triphone-dev-clean.item ./assets/hubert
```

## Reproduce figures from the paper

This saves figures to PDF files:

```bash
uv run gaussians.py
uv run correlation.py
uv run phoneme_vs_speaker.py
```

## Citation

A preprint is available on arXiv: https://arxiv.org/abs/2505.02692 \
If you use fastabx in your work, please cite it:

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
