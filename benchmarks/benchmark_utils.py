"""Shared helpers to run benchmarks and collect results into a single JSON file."""

import json
import statistics
import threading
import time
from pathlib import Path

import psutil


def subset_from_item(item):
    stem = Path(item).stem
    prefix = "triphone-"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


class PeakRSSSampler:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.peak = None
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process()

    def _rss(self):
        total = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            total += child.memory_info().rss
        return total

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak or 0, self._rss())
            self._stop.wait(self.interval)

    def __enter__(self):
        if self._proc is not None:
            self.peak = self._rss()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def run_benchmark(fn, *, with_cuda, runs=1):
    torch = None
    if with_cuda:
        import torch

        track_cuda = torch.cuda.is_available()
    times, peak_cpu, peak_cuda, score = [], None, None, None
    for _ in range(runs):
        if track_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        with PeakRSSSampler() as sampler:
            start = time.perf_counter()
            score = fn()
            if track_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        times.append(elapsed)
        if sampler.peak is not None:
            peak_cpu = max(peak_cpu or 0, sampler.peak)
        if track_cuda:
            peak_cuda = max(peak_cuda or 0, torch.cuda.max_memory_allocated())
    stats = {
        "device": "cuda" if with_cuda else "cpu",
        "runs": runs,
        "time_seconds_mean": statistics.mean(times),
        "time_seconds_all": times,
        "peak_cpu_memory_bytes": peak_cpu,
        "peak_cuda_memory_bytes": peak_cuda,
    }
    return score, stats


def write_result(output, benchmark, item, score, stats):
    output = Path(output)
    subset = subset_from_item(item)
    record = {"benchmark": benchmark, "subset": subset, "score": score, **stats}
    results = []
    if output.exists():
        results = json.loads(output.read_text())
    results = [r for r in results if (r["benchmark"], r["subset"]) != (benchmark, subset)]
    results.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
