# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.7.1",
#     "torchaudio>=2.7.1",
#     "tqdm>=4.67.1",
# ]
# ///
import argparse
from pathlib import Path

import torch
import torchaudio
from torch.nn import functional as F
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", type=Path, help="Path to audio files")
    parser.add_argument("dest", type=Path, help="Path to the destination directory")
    parser.add_argument("--extension", type=str, default=".flac", help="File extension")
    parser.add_argument("--model", type=str, default="HUBERT_BASE", help="Name of the model in torchaudio.pipelines")
    parser.add_argument("--layer", type=int, default=11, help="Layer to use")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = getattr(torchaudio.pipelines, args.model.upper())
    model = bundle.get_model().eval().to(device)

    args.dest.mkdir(exist_ok=True, parents=True)
    for path in tqdm(list(args.root.rglob(f"*{args.extension}"))):
        x, sr = torchaudio.load(str(path))
        assert sr == bundle.sample_rate
        x = F.layer_norm(x, x.shape)  # Normalize audio
        features, _ = model.extract_features(x.to(device))
        torch.save(features[args.layer - 1].squeeze().cpu(), args.dest / f"{path.stem}.pt")
