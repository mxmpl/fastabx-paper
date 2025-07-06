# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastabx>=0.4.1",
#     "torch>=2.7.1",
#     "torchaudio>=2.7.1",
# ]
# ///
import argparse
import tempfile
from pathlib import Path

import torch
import torchaudio
from fastabx import Dataset, Score, Task
from torchaudio.compliance.kaldi import mfcc

SAMPLE_RATE = 16_000
HALF_FRAME = 0.0125  # Half frame length in seconds (=25ms / 2)
FRAME_SHIFT = 0.01  # Frame shift in seconds (=10ms)


def build_mfcc(root: Path, features: Path, times: Path, extension: str) -> None:
    features.mkdir(exist_ok=True)
    times.mkdir(exist_ok=True)
    for path in root.rglob("*.flac"):
        audio, sr = torchaudio.load(str(path))
        assert sr == SAMPLE_RATE
        m = mfcc(audio)
        t = torch.linspace(HALF_FRAME, (len(m) - 1) * FRAME_SHIFT + HALF_FRAME, len(m), dtype=torch.float64)
        torch.save(m, features / f"{path.stem}.pt")
        torch.save(t, times / f"{path.stem}.pt")


def compute_abx(item: Path, features: Path, times: Path) -> None:
    dataset = Dataset.from_item_with_times(item, features, times)
    task_phoneme = Task(dataset, on="#phone", by=["next-phone", "prev-phone", "speaker"])
    score_phoneme = Score(task_phoneme, "angular").collapse(levels=[("next-phone", "prev-phone"), "speaker"])
    print("Phoneme: ", score_phoneme)
    task_speaker = Task(dataset, on="speaker", by=["next-phone", "prev-phone", "#phone"])
    score_speaker = Score(task_speaker, "angular").collapse(levels=[("next-phone", "prev-phone"), "#phone"])
    print("Speaker: ", score_speaker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Path to audio files")
    parser.add_argument("item", type=Path, help="Path to the item file")
    parser.add_argument("--extension", type=str, default=".flac", help="Audio extension (default: .flac)")
    parser.add_argument("--tempdir", type=Path, default=None)
    args = parser.parse_args()
    if args.tempdir is not None:
        args.tempdir.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=args.tempdir) as tempname:
        tempdir = Path(tempname)
        build_mfcc(args.root, tempdir / "features", tempdir / "times", args.extension)
        compute_abx(args.item, tempdir / "features", tempdir / "times")
