"""End-to-end mojo-soundfile benchmarks against upstream soundfile."""

from __future__ import annotations

import math
import os
import platform
import sys
import tempfile
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojosoundfile as msf  # noqa: E402
import soundfile as sf  # noqa: E402


def timeit(fn, repeat=5):
    fn()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    rng = np.random.default_rng(7)
    wav_data = np.ascontiguousarray(rng.uniform(-0.98, 0.98, size=(1_000_000, 2)))
    flac_data = wav_data[:300_000]
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        wav16 = os.path.join(directory, "source16.wav")
        wav24 = os.path.join(directory, "source24.wav")
        wav_float = os.path.join(directory, "source-float.wav")
        flac24 = os.path.join(directory, "source.flac")
        sf.write(wav16, wav_data, 48000, subtype="PCM_16")
        sf.write(wav24, wav_data, 48000, subtype="PCM_24")
        sf.write(wav_float, wav_data, 48000, subtype="FLOAT")
        sf.write(flac24, flac_data, 48000, subtype="PCM_24")

        cases = [
            (
                "WAV PCM_16 read, 1M x 2",
                lambda: msf.read(wav16),
                lambda: sf.read(wav16),
            ),
            (
                "WAV PCM_24 read, 1M x 2",
                lambda: msf.read(wav24),
                lambda: sf.read(wav24),
            ),
            (
                "WAV FLOAT read, 1M x 2",
                lambda: msf.read(wav_float),
                lambda: sf.read(wav_float),
            ),
            (
                "WAV PCM_16 write, 1M x 2",
                lambda: msf.write(os.path.join(directory, "ours16.wav"), wav_data, 48000),
                lambda: sf.write(os.path.join(directory, "sf16.wav"), wav_data, 48000),
            ),
            (
                "WAV PCM_24 write, 1M x 2",
                lambda: msf.write(
                    os.path.join(directory, "ours24.wav"),
                    wav_data,
                    48000,
                    subtype="PCM_24",
                ),
                lambda: sf.write(
                    os.path.join(directory, "sf24.wav"),
                    wav_data,
                    48000,
                    subtype="PCM_24",
                ),
            ),
            (
                "FLAC PCM_24 read, 300k x 2",
                lambda: msf.read(flac24),
                lambda: sf.read(flac24),
            ),
            (
                "FLAC PCM_24 write, 300k x 2",
                lambda: msf.write(
                    os.path.join(directory, "ours.flac"),
                    flac_data,
                    48000,
                    subtype="PCM_24",
                ),
                lambda: sf.write(
                    os.path.join(directory, "sf.flac"),
                    flac_data,
                    48000,
                    subtype="PCM_24",
                ),
            ),
        ]
        for name, ours, upstream in cases:
            ours_time = timeit(ours)
            upstream_time = timeit(upstream)
            ratio = upstream_time / ours_time
            result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
            rows.append((name, ours_time * 1000, upstream_time * 1000, result))

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; Python {platform.python_version()}")
    print()
    print("| case | mojo-soundfile | soundfile | result |")
    print("|---|---:|---:|---:|")
    for name, ours_ms, upstream_ms, result in rows:
        print(f"| {name} | {ours_ms:.2f} ms | {upstream_ms:.2f} ms | {result} |")


if __name__ == "__main__":
    main()
