# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "h5features>=2.0.0",
#     "torch>=2.12.1",
#     "torchaudio>=2.11.0",
#     "torchcodec>=0.14.0",
#     "tqdm>=4.68.3",
# ]
# ///
import argparse
from pathlib import Path

import h5features
import torch
import torchaudio
from torch.nn import functional as F
from torchcodec.decoders import AudioDecoder, WavDecoder
from tqdm import tqdm
import numpy as np


def get_times(length: int, step: float) -> np.ndarray:
    return np.arange(step / 2, length * step, step, dtype=np.float64)


@torch.inference_mode()
def extract_hubert_features(
    path_audio: Path,
    path_features: Path,
    path_archive: Path,
    *,
    layer: int,
    extension: str,
) -> None:
    path_features.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model().eval().to(device)
    decoder = WavDecoder if extension == ".wav" else AudioDecoder
    writer = h5features.Writer(path_archive)

    for path in tqdm(list(path_audio.rglob(f"*{extension}"))):
        samples = decoder(path).get_all_samples()
        assert samples.sample_rate == bundle.sample_rate
        x = F.layer_norm(samples.data, samples.data.shape)
        all_features, _ = model.extract_features(x.to(device))  # ty:ignore[call-non-callable]
        features = all_features[layer - 1].squeeze().cpu()
        torch.save(features, path_features / f"{path.stem}.pt")
        writer.write(h5features.Item(path.stem, features.numpy(), get_times(features.size(0), 0.02)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path_audio", type=Path, help="Path to audio files")
    parser.add_argument("path_features", type=Path, help="Path to the destination directory")
    parser.add_argument("path_archive", type=Path, help="Path to the destination HDF5 archive")
    parser.add_argument("--layer", type=int, default=11, help="Layer to use")
    parser.add_argument("--extension", type=str, default=".flac", help="File extension")
    args = parser.parse_args()
    extract_hubert_features(
        args.path_audio,
        args.path_features,
        args.path_archive,
        layer=args.layer,
        extension=args.extension,
    )
