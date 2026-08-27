# mojo-soundfile

`mojo-soundfile` is a Python package for PCM/WAV and FLAC audio I/O.
Its API follows the corresponding part of
[`soundfile`](https://python-soundfile.readthedocs.io/), while the hot WAV
sample conversion and quantization loops are compiled from Mojo. FLAC is
encoded and decoded through a small direct `ctypes` binding to libsndfile; the
upstream Python package is only used by the tests and benchmarks.

## Coverage

Covered containers and encodings:

- WAV: `PCM_U8`, `PCM_16`, `PCM_24`, `PCM_32`, `FLOAT`, and `DOUBLE`
- FLAC: `PCM_S8`, `PCM_16`, and `PCM_24`
- NumPy data types: `float64`, `float32`, `int32`, and `int16`
- Top-level `read()`, `write()`, `blocks()`, `info()`,
  `available_formats()`, `available_subtypes()`, `default_subtype()`, and
  `check_format()`
- Stateful `SoundFile` reading, writing, seeking, buffer I/O, context-manager
  use, and the tested `title` and `artist` text metadata fields
- Path, file-descriptor, and file-like input; file-like objects are staged
  through a temporary file for libsndfile operations

Not covered are AIFF, Ogg, MP3, RAW streams, RF64/W64/WAVEX, WAV ADPCM,
mu-law/A-law, cues, broadcast metadata, bitrate modes, or the less common
`SoundFile` commands. The package deliberately reports only formats and
subtypes it supports.

## Install

From a source checkout, the supplied Pixi environment installs the pinned Mojo nightly, Python,
NumPy, libsndfile, upstream `soundfile` for parity testing, and pytest:

```bash
pixi install
pixi run build
pixi run test
```

The shared library is written to `dist/libmojo-soundfile.so`. This project
currently supports Linux source-checkout installs through Pixi; it does not
yet publish wheels or bundle libsndfile.

## Usage

```python
import numpy as np
import mojosoundfile as sf

rate = 48_000
t = np.arange(rate, dtype=np.float64) / rate
tone = 0.25 * np.sin(2 * np.pi * 440 * t)

sf.write("tone.wav", tone, rate, subtype="PCM_24")
samples, sample_rate = sf.read("tone.wav", dtype="float64")

sf.write("tone.flac", samples, sample_rate, subtype="PCM_24")
print(sf.info("tone.flac"))
```

Save it as `example.py` in the repository root and run it inside the
environment so `python/mojosoundfile` is on `PYTHONPATH`:

```bash
pixi run python example.py
```

## Benchmark

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, and Python 3.13.14. Times are the best of five warm
runs and include file I/O, container handling, allocation, and conversion.

| case | mojo-soundfile | soundfile | result |
|---|---:|---:|---:|
| WAV PCM_16 read, 1M x 2 | 1.99 ms | 2.66 ms | 1.33x faster |
| WAV PCM_24 read, 1M x 2 | 2.29 ms | 6.18 ms | 2.69x faster |
| WAV FLOAT read, 1M x 2 | 3.30 ms | 5.14 ms | 1.56x faster |
| WAV PCM_16 write, 1M x 2 | 7.26 ms | 18.91 ms | 2.60x faster |
| WAV PCM_24 write, 1M x 2 | 8.88 ms | 24.26 ms | 2.73x faster |
| FLAC PCM_24 read, 300k x 2 | 10.71 ms | 10.44 ms | 1.03x slower |
| FLAC PCM_24 write, 300k x 2 | 21.15 ms | 27.20 ms | 1.29x faster |

These are the literal results of the final `pixi run bench` publication run,
not isolated kernel timings. FLAC PCM_24 read was 1.03x slower than upstream
in this run; performance will vary by machine and workload.

## How it works

`src/soundfile.mojo` is one compilation unit containing byte-to-sample and
sample-to-byte kernels for the covered little-endian PCM and IEEE formats.
The conversion loops use the host's native float64 SIMD width with scalar
remainder handling. PCM_24 decoding expands packed three-byte samples with
contiguous SIMD loads and byte shuffles. Very large inputs are split into
independent chunks across a bounded number of CPU workers; smaller inputs stay
serial to avoid thread-launch overhead.

Python owns all input and output memory. Buffers cross the C ABI as integer
addresses, and the exported Mojo functions reconstruct
`UnsafePointer[..., AnyOrigin[mut=True]]` values internally. Samples are
interleaved frame-major in C-contiguous NumPy arrays: `(frames,)` for mono by
default and `(frames, channels)` for multichannel data. Path-based WAV reads
memory-map the source, and WAV writes pass the NumPy output buffer directly
to the file object without an intermediate `bytes` copy.

There is no GPU path. PCM conversion and packing have low arithmetic
intensity (well below two floating-point operations per byte moved), so
device transfer and launch overhead would dominate.

The Python WAV layer parses and emits RIFF chunks, then calls the Mojo shared
library for conversion. Integer-output reads, stateful access, and FLAC use
libsndfile directly through `ctypes`, without importing upstream `soundfile`.
The tests generate files in both implementations and cross-read them, so they
assert exact sample, dtype, shape, slicing, and metadata parity rather than
only checking that calls complete.
