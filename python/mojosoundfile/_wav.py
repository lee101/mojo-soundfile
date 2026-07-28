"""RIFF/WAV container handling around the Mojo PCM kernels."""

from __future__ import annotations

import io
import mmap
import os
import struct
from dataclasses import dataclass

import numpy as np

from ._mojo import decode_f64, encode_f64

_KINDS = {
    "PCM_U8": (1, 1, 1),
    "PCM_16": (2, 2, 1),
    "PCM_24": (3, 3, 1),
    "PCM_32": (4, 4, 1),
    "FLOAT": (6, 4, 3),
    "DOUBLE": (7, 8, 3),
}


class UnsupportedWavError(ValueError):
    """The WAV is valid enough to inspect but its encoding is not supported."""


class NotWavError(ValueError):
    """The input does not have a RIFF/WAVE signature."""


@dataclass(frozen=True)
class WavHeader:
    samplerate: int
    channels: int
    frames: int
    subtype: str
    data_offset: int
    data_size: int


def _open_bytes(file) -> bytes:
    if isinstance(file, (str, bytes, os.PathLike)):
        with open(file, "rb") as stream:
            return stream.read()
    if not hasattr(file, "read"):
        raise TypeError(f"Invalid file: {file!r}")
    position = file.tell() if hasattr(file, "tell") else None
    payload = file.read()
    if position is not None and hasattr(file, "seek"):
        file.seek(position)
    return payload


def parse(payload: bytes) -> WavHeader:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise NotWavError("not a little-endian RIFF/WAVE file")
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(payload):
        chunk_id = payload[pos : pos + 4]
        size = struct.unpack_from("<I", payload, pos + 4)[0]
        start = pos + 8
        if start + size > len(payload):
            raise ValueError("truncated WAV chunk")
        if chunk_id == b"fmt ":
            fmt = payload[start : start + size]
        elif chunk_id == b"data" and data is None:
            data = (start, size)
        pos = start + size + (size & 1)
    if fmt is None or data is None or len(fmt) < 16:
        raise ValueError("WAV file is missing fmt or data chunk")
    code, channels, samplerate, _, blockalign, bits = struct.unpack_from("<HHIIHH", fmt)
    if code == 0xFFFE and len(fmt) >= 40:
        code = struct.unpack_from("<H", fmt, 24)[0]
    subtype = {
        (1, 8): "PCM_U8",
        (1, 16): "PCM_16",
        (1, 24): "PCM_24",
        (1, 32): "PCM_32",
        (3, 32): "FLOAT",
        (3, 64): "DOUBLE",
    }.get((code, bits))
    if subtype is None or channels < 1 or blockalign != channels * (bits // 8):
        raise UnsupportedWavError("unsupported WAV encoding")
    if data[1] % blockalign:
        raise ValueError("WAV data chunk is not a whole number of frames")
    frames = data[1] // blockalign
    return WavHeader(samplerate, channels, frames, subtype, data[0], data[1])


def can_fast_read(file, dtype: str, format: str | None) -> bool:
    if dtype not in ("float64", "float32"):
        return False
    if format not in (None, "WAV", "wav"):
        return False
    if isinstance(file, (str, bytes, os.PathLike)):
        return os.fspath(file).lower().endswith(".wav")
    return isinstance(file, (io.BytesIO, io.BufferedIOBase))


def read(
    file,
    frames: int,
    start: int,
    stop: int | None,
    dtype: str,
    always_2d: bool,
    fill_value,
    out: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    stream = None
    mapping = None
    if isinstance(file, (str, bytes, os.PathLike)):
        stream = open(file, "rb")
        try:
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            stream.close()
            raise
        payload = mapping
    else:
        payload = _open_bytes(file)
    try:
        header = parse(payload)
        if frames >= 0 and stop is not None:
            raise TypeError("Only one of {frames, stop} may be used")
        begin, end, _ = slice(start, stop).indices(header.frames)
        if end < begin:
            end = begin
        requested = end - begin if frames < 0 else frames
        available = max(0, min(requested, header.frames - begin))
        kind, width, _ = _KINDS[header.subtype]
        byte_start = header.data_offset + begin * header.channels * width
        byte_end = byte_start + available * header.channels * width
        segment = memoryview(payload)[byte_start:byte_end]
        try:
            values = decode_f64(segment, available * header.channels, kind)
        finally:
            segment.release()
    finally:
        if mapping is not None:
            mapping.close()
            stream.close()
    values = values.reshape(available, header.channels)
    if dtype == "float32":
        values = values.astype(np.float32)
    if out is not None:
        target = np.asarray(out)
        valid_shape = (
            target.ndim == 1 and header.channels == 1
        ) or (
            target.ndim == 2 and target.shape[1] == header.channels
        )
        if (
            target.dtype.name != dtype
            or not target.flags.c_contiguous
            or not target.flags.writeable
            or not target.dtype.isnative
            or not valid_shape
        ):
            raise ValueError(
                "out must be writable, C-contiguous, match dtype, and have shape "
                "(frames,) for mono or (frames, channels)"
            )
        limit = min(target.shape[0], requested)
        copied = min(available, limit)
        if target.ndim == 1:
            if header.channels != 1:
                raise ValueError("out has wrong number of channels")
            target[:copied] = values[:copied, 0]
        else:
            if target.shape[1] != header.channels:
                raise ValueError("out has wrong number of channels")
            target[:copied] = values[:copied]
        if copied < limit and fill_value is not None:
            target[copied:limit] = fill_value
            copied = limit
        return target[:copied], header.samplerate
    if available < requested and fill_value is not None:
        filled = np.full((requested, header.channels), fill_value, dtype=dtype)
        filled[:available] = values
        values = filled
    if header.channels == 1 and not always_2d:
        values = values[:, 0]
    return values, header.samplerate


def write(file, data, samplerate: int, subtype: str | None) -> None:
    subtype = "PCM_16" if subtype is None else subtype.upper()
    if subtype not in _KINDS:
        raise ValueError(f"Unsupported WAV subtype: {subtype!r}")
    values = np.asarray(data)
    if values.ndim == 1:
        channels = 1
    elif values.ndim == 2 and values.shape[1] > 0:
        channels = values.shape[1]
    else:
        raise ValueError("Invalid shape: audio data must be one- or two-dimensional")
    kind, width, code = _KINDS[subtype]
    if values.dtype.name in ("int16", "int32"):
        raise TypeError("integer input uses the libsndfile path")
    raw = encode_f64(np.ascontiguousarray(values, dtype=np.float64), kind, width)
    byte_rate = int(samplerate) * channels * width
    block_align = channels * width
    bits = width * 8
    pad = bool(len(raw) & 1)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(raw) + pad)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, code, channels, int(samplerate), byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", len(raw))
    )
    if isinstance(file, (str, bytes, os.PathLike)):
        with open(file, "wb") as stream:
            stream.write(header)
            stream.write(raw)
            if pad:
                stream.write(b"\0")
    else:
        file.write(header)
        file.write(raw)
        if pad:
            file.write(b"\0")
