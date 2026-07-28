"""ctypes bridge to the Mojo PCM conversion kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "soundfile.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-soundfile.so")

I = ctypes.c_int64
_SIGNATURES = {
    "msf_decode_f64": ([I, I, I, I], None),
    "msf_encode_f64": ([I, I, I, I], None),
}
_KIND_WIDTHS = {1: 1, 2: 2, 3: 3, 4: 4, 6: 4, 7: 8}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run inside `pixi run`")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIB, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIB


def decode_f64(raw: bytes | bytearray | memoryview, count: int, kind: int) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    if kind not in _KIND_WIDTHS:
        raise ValueError(f"unsupported sample kind: {kind!r}")
    view = memoryview(raw)
    required = count * _KIND_WIDTHS[kind]
    if not view.c_contiguous:
        raise ValueError("raw buffer must be C-contiguous")
    if view.nbytes < required:
        raise ValueError(
            f"raw buffer is too short: need {required} bytes, got {view.nbytes}"
        )
    buf = np.frombuffer(view, dtype=np.uint8, count=required)
    result = np.empty(count, dtype=np.float64)
    if count:
        lib().msf_decode_f64(buf.ctypes.data, result.ctypes.data, count, kind)
    return result


def encode_f64(data: np.ndarray, kind: int, width: int) -> np.ndarray:
    if kind not in _KIND_WIDTHS:
        raise ValueError(f"unsupported sample kind: {kind!r}")
    if width != _KIND_WIDTHS[kind]:
        raise ValueError(
            f"sample kind {kind} requires width {_KIND_WIDTHS[kind]}, got {width}"
        )
    values = np.ascontiguousarray(data, dtype=np.float64).reshape(-1)
    raw = np.empty(values.size * width, dtype=np.uint8)
    if values.size:
        lib().msf_encode_f64(values.ctypes.data, raw.ctypes.data, values.size, kind)
    return raw
