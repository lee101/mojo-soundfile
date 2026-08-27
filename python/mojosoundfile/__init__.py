"""SoundFile-compatible WAV/FLAC I/O with Mojo PCM kernels."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import _wav
from ._sndfile import (
    LibsndfileError,
    SoundFile,
    available_formats,
    available_subtypes,
    check_format,
    default_subtype,
)

__version__ = "0.1.0"
SEEK_SET, SEEK_CUR, SEEK_END = os.SEEK_SET, os.SEEK_CUR, os.SEEK_END


def read(
    file,
    frames: int = -1,
    start: int = 0,
    stop: int | None = None,
    dtype: str = "float64",
    always_2d: bool = False,
    fill_value=None,
    out: np.ndarray | None = None,
    samplerate: int | None = None,
    channels: int | None = None,
    format: str | None = None,
    subtype: str | None = None,
    endian: str | None = None,
    closefd: bool = True,
) -> tuple[np.ndarray, int]:
    """Read audio data and its sample rate.

    Common little-endian WAV files with floating-point output use the Mojo
    conversion path. FLAC and other covered combinations go directly through
    libsndfile.
    """
    if dtype not in ("float64", "float32", "int32", "int16"):
        raise ValueError("dtype must be one of {'float64', 'float32', 'int32', 'int16'}")
    if (
        samplerate is None
        and channels is None
        and subtype is None
        and endian is None
        and _wav.can_fast_read(file, dtype, format)
    ):
        try:
            return _wav.read(file, frames, start, stop, dtype, always_2d, fill_value, out)
        except (_wav.NotWavError, _wav.UnsupportedWavError):
            pass
    with SoundFile(
        file,
        "r",
        samplerate=samplerate,
        channels=channels,
        format=format,
        subtype=subtype,
        endian=endian,
        closefd=closefd,
    ) as stream:
        if frames >= 0 and stop is not None:
            raise TypeError("Only one of {frames, stop} may be used")
        begin, end, _ = slice(start, stop).indices(stream.frames)
        if end < begin:
            end = begin
        requested = end - begin if frames < 0 else frames
        if begin and stream.seekable():
            stream.seek(begin)
        data = stream.read(
            requested,
            dtype=dtype,
            always_2d=always_2d,
            fill_value=fill_value,
            out=out,
        )
        return data, stream.samplerate


def write(
    file,
    data,
    samplerate: int,
    subtype: str | None = None,
    endian: str | None = None,
    format: str | None = None,
    closefd: bool = True,
    compression_level: float | None = None,
    bitrate_mode: str | None = None,
) -> None:
    """Write a one- or two-dimensional audio array."""
    name = getattr(file, "name", file)
    inferred = ""
    if isinstance(name, (str, bytes, os.PathLike)):
        inferred = os.path.splitext(os.fsdecode(name))[1][1:].upper()
    fmt = inferred if format is None else format.upper()
    arr = np.asarray(data)
    if arr.dtype.name not in ("float64", "float32", "int32", "int16"):
        raise ValueError(
            "dtype must be one of {'float64', 'float32', 'int32', 'int16'}, "
            f"not {arr.dtype.name!r}"
        )
    if (
        fmt == "WAV"
        and arr.dtype.name not in ("int16", "int32")
        and (endian is None or endian.upper() in ("FILE", "LITTLE"))
        and compression_level is None
        and bitrate_mode is None
    ):
        _wav.write(file, arr, samplerate, subtype)
        return
    channels = 1 if arr.ndim == 1 else (arr.shape[1] if arr.ndim == 2 else None)
    if channels is None:
        raise ValueError("Invalid shape: audio data must be one- or two-dimensional")
    with SoundFile(
        file,
        "w",
        samplerate=samplerate,
        channels=channels,
        subtype=subtype,
        endian=endian,
        format=format,
        closefd=closefd,
        compression_level=compression_level,
        bitrate_mode=bitrate_mode,
    ) as stream:
        stream.write(arr)


def blocks(
    file,
    blocksize: int | None = None,
    overlap: int = 0,
    frames: int = -1,
    start: int = 0,
    stop: int | None = None,
    dtype: str = "float64",
    always_2d: bool = False,
    fill_value=None,
    out: np.ndarray | None = None,
    samplerate: int | None = None,
    channels: int | None = None,
    format: str | None = None,
    subtype: str | None = None,
    endian: str | None = None,
    closefd: bool = True,
):
    """Yield overlapping blocks, matching the covered upstream interface."""
    if blocksize is None and out is None:
        raise TypeError("One of {blocksize, out} must be specified")
    size = out.shape[0] if blocksize is None else int(blocksize)
    if size <= 0:
        raise ValueError("blocksize must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < blocksize")
    with SoundFile(
        file,
        "r",
        samplerate=samplerate,
        channels=channels,
        format=format,
        subtype=subtype,
        endian=endian,
        closefd=closefd,
    ) as stream:
        if frames >= 0 and stop is not None:
            raise TypeError("Only one of {frames, stop} may be used")
        begin, end, _ = slice(start, stop).indices(stream.frames)
        remaining = end - begin if frames < 0 else min(frames, stream.frames - begin)
        stream.seek(begin)
        while remaining > 0:
            count = min(size, remaining)
            target = out if out is not None else None
            block = stream.read(
                count,
                dtype=dtype,
                always_2d=always_2d,
                fill_value=fill_value,
                out=target,
            )
            if not len(block):
                break
            yield block
            if remaining <= size or len(block) < count:
                break
            advance = len(block) - overlap
            remaining -= advance
            if overlap and remaining > 0:
                stream.seek(-overlap, SEEK_CUR)


@dataclass(frozen=True)
class _SoundFileInfo:
    name: object
    samplerate: int
    channels: int
    frames: int
    format: str
    subtype: str
    endian: str
    sections: int
    seekable: bool
    duration: float
    extra_info: str = ""

    def __str__(self) -> str:
        return (
            f"{self.name}\n"
            f"samplerate: {self.samplerate} Hz\n"
            f"channels: {self.channels}\n"
            f"duration: {self.duration:.3f} s\n"
            f"format: {self.format}\n"
            f"subtype: {self.subtype}"
        )


def info(file, verbose: bool = False) -> _SoundFileInfo:
    """Return immutable basic file information."""
    with SoundFile(file) as stream:
        return _SoundFileInfo(
            stream.name,
            stream.samplerate,
            stream.channels,
            stream.frames,
            stream.format,
            stream.subtype,
            stream.endian,
            stream.sections,
            stream.seekable(),
            stream.frames / stream.samplerate,
            stream.extra_info if verbose else "",
        )


__all__ = [
    "LibsndfileError",
    "SEEK_CUR",
    "SEEK_END",
    "SEEK_SET",
    "SoundFile",
    "available_formats",
    "available_subtypes",
    "blocks",
    "check_format",
    "default_subtype",
    "info",
    "read",
    "write",
]
