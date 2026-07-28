"""Small ctypes binding to libsndfile for FLAC and stateful file access."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import tempfile
from pathlib import Path

import numpy as np

_FORMATS = {"WAV": 0x010000, "FLAC": 0x170000}
_FORMAT_NAMES = {
    "WAV": "WAV (Microsoft)",
    "FLAC": "FLAC (Free Lossless Audio Codec)",
}
_SUBTYPES = {
    "PCM_S8": 0x0001,
    "PCM_16": 0x0002,
    "PCM_24": 0x0003,
    "PCM_32": 0x0004,
    "PCM_U8": 0x0005,
    "FLOAT": 0x0006,
    "DOUBLE": 0x0007,
}
_SUBTYPE_NAMES = {
    "PCM_S8": "Signed 8 bit PCM",
    "PCM_16": "Signed 16 bit PCM",
    "PCM_24": "Signed 24 bit PCM",
    "PCM_32": "Signed 32 bit PCM",
    "PCM_U8": "Unsigned 8 bit PCM",
    "FLOAT": "32 bit float",
    "DOUBLE": "64 bit float",
}
_ENDIANS = {"FILE": 0, "LITTLE": 0x10000000, "BIG": 0x20000000, "CPU": 0x30000000}
_DEFAULTS = {"WAV": "PCM_16", "FLAC": "PCM_16"}
_FORMAT_MASK = 0x0FFF0000
_SUBTYPE_MASK = 0x0000FFFF
_ENDIAN_MASK = 0x30000000
_SFM_READ, _SFM_WRITE, _SFM_RDWR = 0x10, 0x20, 0x30


class _SFInfo(ctypes.Structure):
    _fields_ = [
        ("frames", ctypes.c_int64),
        ("samplerate", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("format", ctypes.c_int),
        ("sections", ctypes.c_int),
        ("seekable", ctypes.c_int),
    ]


def _load() -> ctypes.CDLL:
    name = (
        os.environ.get("MOJOSOUNDFILE_LIBSNDFILE")
        or ctypes.util.find_library("sndfile")
        or "libsndfile.so.1"
    )
    dll = ctypes.CDLL(name)
    ptr = ctypes.c_void_p
    dll.sf_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(_SFInfo)]
    dll.sf_open.restype = ptr
    dll.sf_open_fd.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_SFInfo), ctypes.c_int]
    dll.sf_open_fd.restype = ptr
    dll.sf_close.argtypes = [ptr]
    dll.sf_close.restype = ctypes.c_int
    dll.sf_error.argtypes = [ptr]
    dll.sf_error.restype = ctypes.c_int
    dll.sf_strerror.argtypes = [ptr]
    dll.sf_strerror.restype = ctypes.c_char_p
    dll.sf_format_check.argtypes = [ctypes.POINTER(_SFInfo)]
    dll.sf_format_check.restype = ctypes.c_int
    dll.sf_command.argtypes = [ptr, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    dll.sf_command.restype = ctypes.c_int
    dll.sf_seek.argtypes = [ptr, ctypes.c_int64, ctypes.c_int]
    dll.sf_seek.restype = ctypes.c_int64
    dll.sf_write_sync.argtypes = [ptr]
    for suffix, ctype in (
        ("double", ctypes.c_double),
        ("float", ctypes.c_float),
        ("int", ctypes.c_int32),
        ("short", ctypes.c_int16),
    ):
        for op in ("readf", "writef"):
            fn = getattr(dll, f"sf_{op}_{suffix}")
            fn.argtypes = [ptr, ctypes.POINTER(ctype), ctypes.c_int64]
            fn.restype = ctypes.c_int64
    dll.sf_get_string.argtypes = [ptr, ctypes.c_int]
    dll.sf_get_string.restype = ctypes.c_char_p
    dll.sf_set_string.argtypes = [ptr, ctypes.c_int, ctypes.c_char_p]
    dll.sf_set_string.restype = ctypes.c_int
    return dll


_LIB = _load()


class LibsndfileError(RuntimeError):
    def __init__(self, code: int, prefix: str = ""):
        self.code = int(code)
        self.error_string = prefix + (_LIB.sf_strerror(None) or b"unknown error").decode(
            "utf-8", "replace"
        )
        super().__init__(self.error_string)


def _error(handle, prefix: str) -> LibsndfileError:
    message = (_LIB.sf_strerror(handle) or b"unknown error").decode("utf-8", "replace")
    err = LibsndfileError.__new__(LibsndfileError)
    err.code = int(_LIB.sf_error(handle))
    err.error_string = prefix + message
    RuntimeError.__init__(err, err.error_string)
    return err


def available_formats() -> dict[str, str]:
    return dict(_FORMAT_NAMES)


def available_subtypes(format: str | None = None) -> dict[str, str]:
    if format is None:
        return dict(_SUBTYPE_NAMES)
    fmt = _check_format(format)
    allowed = (
        ("PCM_S8", "PCM_16", "PCM_24")
        if fmt == "FLAC"
        else ("PCM_16", "PCM_24", "PCM_32", "PCM_U8", "FLOAT", "DOUBLE")
    )
    return {key: _SUBTYPE_NAMES[key] for key in allowed}


def default_subtype(format: str) -> str | None:
    return _DEFAULTS[_check_format(format)]


def _check_format(format: str) -> str:
    if not isinstance(format, str):
        raise TypeError(f"Invalid format: {format!r}")
    result = format.upper()
    if result not in _FORMATS:
        raise ValueError(f"Unknown format: {format!r}")
    return result


def _format_int(format: str, subtype: str | None, endian: str | None) -> int:
    fmt = _check_format(format)
    subtype = default_subtype(fmt) if subtype is None else subtype.upper()
    endian = "FILE" if endian is None else endian.upper()
    if subtype not in _SUBTYPES:
        raise ValueError(f"Unknown subtype: {subtype!r}")
    if endian not in _ENDIANS:
        raise ValueError(f"Unknown endian-ness: {endian!r}")
    info = _SFInfo(channels=1, format=_FORMATS[fmt] | _SUBTYPES[subtype] | _ENDIANS[endian])
    if not _LIB.sf_format_check(ctypes.byref(info)):
        raise ValueError("Invalid combination of format, subtype and endian")
    return info.format


def check_format(format: str, subtype: str | None = None, endian: str | None = None) -> bool:
    try:
        _format_int(format, subtype, endian)
    except (TypeError, ValueError):
        return False
    return True


def _format_name(value: int) -> str:
    target = value & _FORMAT_MASK
    return next((k for k, v in _FORMATS.items() if v == target), "n/a")


def _subtype_name(value: int) -> str:
    target = value & _SUBTYPE_MASK
    return next((k for k, v in _SUBTYPES.items() if v == target), "n/a")


def _endian_name(value: int) -> str:
    target = value & _ENDIAN_MASK
    return next((k for k, v in _ENDIANS.items() if v == target), "FILE")


def _mode_int(mode: str) -> int:
    if not isinstance(mode, str):
        raise TypeError(f"Invalid mode: {mode!r}")
    chars = set(mode)
    if chars.difference("xrwb+") or len(mode) > len(chars):
        raise ValueError(f"Invalid mode: {mode!r}")
    if len(chars.intersection("xrw")) != 1:
        raise ValueError("mode must contain exactly one of 'xrw'")
    if "+" in chars:
        return _SFM_RDWR
    return _SFM_READ if "r" in chars else _SFM_WRITE


def _path_format(file, mode: str, explicit: str | None) -> str:
    if explicit is not None:
        return _check_format(explicit)
    name = getattr(file, "name", file)
    suffix = ""
    if isinstance(name, (str, bytes, os.PathLike)):
        suffix = Path(os.fsdecode(name)).suffix[1:].upper()
    if suffix in _FORMATS:
        return suffix
    if "r" not in mode:
        raise TypeError(f"No format specified and unable to get format from file extension: {name!r}")
    return ""


_DTYPES = {
    "float64": (np.dtype("float64"), "double"),
    "float32": (np.dtype("float32"), "float"),
    "int32": (np.dtype("int32"), "int"),
    "int16": (np.dtype("int16"), "short"),
}


class SoundFile:
    def __init__(
        self,
        file,
        mode: str | None = "r",
        samplerate: int | None = None,
        channels: int | None = None,
        subtype: str | None = None,
        endian: str | None = None,
        format: str | None = None,
        closefd: bool = True,
        compression_level: float | None = None,
        bitrate_mode: str | None = None,
    ) -> None:
        mode = "r" if mode is None else mode
        mode_int = _mode_int(mode)
        self.name = getattr(file, "name", file)
        self.mode = mode
        self._file = None
        self._temp_path: str | None = None
        self._file_object = None
        self._copy_back = False
        self._write_position = 0
        fmt = _path_format(file, mode, format)
        self._info = _SFInfo()
        if "r" not in mode or not fmt:
            if "r" not in mode:
                if samplerate is None:
                    raise TypeError("samplerate must be specified")
                if channels is None:
                    raise TypeError("channels must be specified")
            if samplerate is not None:
                self._info.samplerate = int(samplerate)
            if channels is not None:
                self._info.channels = int(channels)
            if fmt:
                self._info.format = _format_int(fmt, subtype, endian)
        elif any(x is not None for x in (samplerate, channels, subtype, endian, format)):
            raise TypeError(
                "Not allowed for existing files: samplerate, channels, format, subtype, endian"
            )
        if compression_level is not None and not 0 <= compression_level <= 1:
            raise ValueError("Compression level must be in range [0..1]")
        if bitrate_mode is not None:
            raise ValueError("bitrate_mode is not supported for WAV/FLAC")

        target = file
        if not isinstance(file, (str, bytes, os.PathLike, int)):
            tmp = tempfile.NamedTemporaryFile(suffix=f".{(fmt or 'wav').lower()}", delete=False)
            self._temp_path = tmp.name
            tmp.close()
            self._file_object = file
            if "r" in mode:
                pos = file.tell() if hasattr(file, "tell") else None
                payload = file.read()
                with open(self._temp_path, "wb") as stream:
                    stream.write(payload)
                if pos is not None and hasattr(file, "seek"):
                    file.seek(pos)
            self._copy_back = "w" in mode or "x" in mode or "+" in mode
            target = self._temp_path

        if isinstance(target, int):
            handle = _LIB.sf_open_fd(target, mode_int, ctypes.byref(self._info), int(closefd))
        else:
            path = os.fsencode(target)
            if "x" in mode and os.path.exists(target):
                raise OSError(f"File exists: {self.name!r}")
            if "w" in mode and "+" in mode and os.path.exists(target):
                with open(target, "wb"):
                    pass
            handle = _LIB.sf_open(path, mode_int, ctypes.byref(self._info))
        if not handle:
            self._cleanup_temp()
            raise _error(None, f"Error opening {self.name!r}: ")
        self._file = handle
        if compression_level is not None:
            level = ctypes.c_double(compression_level)
            ok = _LIB.sf_command(
                self._file, 0x1301, ctypes.byref(level), ctypes.sizeof(level)
            )
            if not ok:
                self.close()
                raise _error(None, f"Error setting compression level {compression_level}: ")

    @property
    def samplerate(self) -> int:
        self._check_open()
        return int(self._info.samplerate)

    @property
    def frames(self) -> int:
        self._check_open()
        return int(self._info.frames)

    @property
    def channels(self) -> int:
        self._check_open()
        return int(self._info.channels)

    @property
    def format(self) -> str:
        self._check_open()
        return _format_name(self._info.format)

    @property
    def subtype(self) -> str:
        self._check_open()
        return _subtype_name(self._info.format)

    @property
    def endian(self) -> str:
        self._check_open()
        return _endian_name(self._info.format)

    @property
    def format_info(self) -> str:
        return _FORMAT_NAMES[self.format]

    @property
    def subtype_info(self) -> str:
        return _SUBTYPE_NAMES[self.subtype]

    @property
    def sections(self) -> int:
        self._check_open()
        return int(self._info.sections)

    @property
    def closed(self) -> bool:
        return self._file is None

    @property
    def extra_info(self) -> str:
        return ""

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("I/O operation on closed file")

    def __len__(self) -> int:
        return self.frames

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else (
            f"mode={self.mode!r}, samplerate={self.samplerate}, channels={self.channels}, "
            f"format={self.format!r}, subtype={self.subtype!r}, endian={self.endian!r}"
        )
        return f"SoundFile({self.name!r}, {state})"

    def seekable(self) -> bool:
        self._check_open()
        return bool(self._info.seekable)

    def seek(self, frames: int, whence: int = os.SEEK_SET) -> int:
        self._check_open()
        position = _LIB.sf_seek(self._file, int(frames), int(whence))
        if position < 0:
            raise _error(self._file, "Error seeking: ")
        self._write_position = int(position)
        return int(position)

    def tell(self) -> int:
        return self.seek(0, os.SEEK_CUR)

    def read(
        self,
        frames: int = -1,
        dtype: str = "float64",
        always_2d: bool = False,
        fill_value=None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_open()
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {set(_DTYPES)}")
        if out is not None:
            arr = np.asarray(out)
            dtype = arr.dtype.name
            valid_shape = (
                arr.ndim == 1 and self.channels == 1
            ) or (
                arr.ndim == 2 and arr.shape[1] == self.channels
            )
            if (
                dtype not in _DTYPES
                or not arr.flags.c_contiguous
                or not arr.flags.writeable
                or not arr.dtype.isnative
                or not valid_shape
            ):
                raise ValueError(
                    "out must be a writable C-contiguous float64/float32/int32/int16 "
                    "array with shape (frames,) for mono or (frames, channels)"
                )
            requested = arr.shape[0]
            if frames >= 0:
                requested = min(requested, frames)
        else:
            requested = self.frames - self.tell() if frames < 0 else int(frames)
            shape = (requested, self.channels) if always_2d or self.channels > 1 else (requested,)
            arr = np.empty(shape, dtype=_DTYPES[dtype][0])
        flat = arr.reshape(-1)
        ctype = np.ctypeslib.as_ctypes_type(arr.dtype)
        fn = getattr(_LIB, f"sf_readf_{_DTYPES[dtype][1]}")
        got = int(fn(self._file, flat.ctypes.data_as(ctypes.POINTER(ctype)), requested))
        if got < 0:
            raise _error(self._file, "Error reading: ")
        if got < requested and fill_value is not None:
            arr.reshape(requested, -1)[got:] = fill_value
            got = requested
        if got != arr.shape[0]:
            arr = arr[:got]
        if self.channels == 1 and always_2d and arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def write(self, data) -> int:
        self._check_open()
        arr = np.asarray(data)
        if arr.dtype.name not in _DTYPES:
            raise ValueError(
                f"dtype must be one of {set(_DTYPES)}, not {arr.dtype.name!r}"
            )
        if arr.ndim not in (1, 2):
            raise ValueError("Invalid shape: audio data must be one- or two-dimensional")
        channels = 1 if arr.ndim == 1 else arr.shape[1]
        if channels != self.channels:
            raise ValueError(
                f"Invalid shape: expected {self.channels} channels, got {channels}"
            )
        arr = np.ascontiguousarray(arr, dtype=_DTYPES[arr.dtype.name][0])
        frames = arr.shape[0]
        ctype = np.ctypeslib.as_ctypes_type(arr.dtype)
        fn = getattr(_LIB, f"sf_writef_{_DTYPES[arr.dtype.name][1]}")
        wrote = int(fn(self._file, arr.ctypes.data_as(ctypes.POINTER(ctype)), frames))
        if wrote != frames:
            raise _error(self._file, "Error writing: ")
        self._write_position += wrote
        self._info.frames = max(self._info.frames, self._write_position)
        return wrote

    def buffer_read(self, frames: int = -1, dtype: str = "float64") -> bytes:
        return self.read(frames, dtype=dtype, always_2d=True).tobytes()

    def buffer_write(self, data, dtype: str) -> int:
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {set(_DTYPES)}")
        arr = np.frombuffer(data, dtype=_DTYPES[dtype][0])
        if arr.size % self.channels:
            raise ValueError("Data size must be a multiple of channel count")
        return self.write(arr.reshape(-1, self.channels))

    def flush(self) -> None:
        self._check_open()
        _LIB.sf_write_sync(self._file)

    def close(self) -> None:
        if self._file is None:
            return
        handle, self._file = self._file, None
        code = _LIB.sf_close(handle)
        try:
            if self._copy_back and self._temp_path and self._file_object is not None:
                with open(self._temp_path, "rb") as stream:
                    payload = stream.read()
                if hasattr(self._file_object, "seek"):
                    self._file_object.seek(0)
                self._file_object.write(payload)
                if hasattr(self._file_object, "truncate"):
                    self._file_object.truncate()
        finally:
            self._cleanup_temp()
        if code:
            raise LibsndfileError(code, "Error closing file: ")

    def _cleanup_temp(self):
        if self._temp_path:
            try:
                os.unlink(self._temp_path)
            except FileNotFoundError:
                pass
            self._temp_path = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_metadata(self, kind: int) -> str | None:
        self._check_open()
        value = _LIB.sf_get_string(self._file, kind)
        return value.decode("utf-8", "replace") if value else None

    def _set_metadata(self, kind: int, value: str | None) -> None:
        self._check_open()
        encoded = None if value is None else str(value).encode()
        code = _LIB.sf_set_string(self._file, kind, encoded)
        if code:
            raise _error(self._file, "Error setting metadata: ")


for _name, _kind in {
    "title": 0x01,
    "copyright": 0x02,
    "software": 0x03,
    "artist": 0x04,
    "comment": 0x05,
    "date": 0x06,
    "album": 0x07,
    "license": 0x08,
    "tracknumber": 0x09,
    "genre": 0x10,
}.items():
    setattr(
        SoundFile,
        _name,
        property(
            lambda self, kind=_kind: self._get_metadata(kind),
            lambda self, value, kind=_kind: self._set_metadata(kind, value),
        ),
    )
