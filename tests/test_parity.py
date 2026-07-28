import io
import os

import numpy as np
import pytest
import soundfile as sf

import mojosoundfile as msf
from mojosoundfile._mojo import decode_f64, encode_f64

rng = np.random.default_rng(2026)


def signal(frames=4097, channels=1):
    t = np.arange(frames) / 48000
    values = 0.71 * np.sin(2 * np.pi * 997 * t) + 0.18 * np.sin(2 * np.pi * 3301 * t)
    if channels == 1:
        return values
    return np.column_stack([values, values[::-1]])


def test_format_queries_match_covered_upstream_subset():
    assert msf.available_formats() == {
        key: sf.available_formats()[key] for key in ("WAV", "FLAC")
    }
    for fmt in ("WAV", "FLAC"):
        upstream = sf.available_subtypes(fmt)
        assert all(upstream[key] == value for key, value in msf.available_subtypes(fmt).items())
        assert msf.default_subtype(fmt) == sf.default_subtype(fmt)


@pytest.mark.parametrize(
    "fmt,subtype,valid",
    [
        ("WAV", "PCM_16", True),
        ("WAV", "FLOAT", True),
        ("FLAC", "PCM_24", True),
        ("FLAC", "FLOAT", False),
        ("BOGUS", None, False),
    ],
)
def test_check_format_parity(fmt, subtype, valid):
    assert msf.check_format(fmt, subtype) is valid
    assert msf.check_format(fmt, subtype) == sf.check_format(fmt, subtype)


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"])
def test_wav_read_write_matches_upstream(tmp_path, subtype):
    data = signal()
    ours = tmp_path / f"ours-{subtype}.wav"
    theirs = tmp_path / f"theirs-{subtype}.wav"
    msf.write(ours, data, 48000, subtype=subtype)
    sf.write(theirs, data, 48000, subtype=subtype)
    ours_by_sf, rate_a = sf.read(ours)
    ours_by_us, rate_b = msf.read(ours)
    theirs_by_sf, _ = sf.read(theirs)
    theirs_by_us, _ = msf.read(theirs)
    assert rate_a == rate_b == 48000
    assert np.array_equal(ours_by_sf, ours_by_us, equal_nan=True)
    assert np.array_equal(theirs_by_sf, theirs_by_us, equal_nan=True)
    assert np.array_equal(ours_by_us, theirs_by_sf, equal_nan=True)


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "PCM_24", "PCM_32"])
def test_pcm_clipping_and_nonfinite_parity(tmp_path, subtype):
    data = np.array([-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf, np.nan])
    ours = tmp_path / "ours.wav"
    theirs = tmp_path / "theirs.wav"
    msf.write(ours, data, 8000, subtype=subtype)
    sf.write(theirs, data, 8000, subtype=subtype)
    assert np.array_equal(msf.read(ours)[0], sf.read(theirs)[0])


@pytest.mark.parametrize("frames", [1, 3, 5, 7])
@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_mojo_simd_tail_matches_upstream(tmp_path, frames, subtype):
    data = signal(frames)
    ours = tmp_path / f"ours-{subtype}-{frames}.wav"
    theirs = tmp_path / f"theirs-{subtype}-{frames}.wav"
    msf.write(ours, data, 48000, subtype=subtype)
    sf.write(theirs, data, 48000, subtype=subtype)
    assert np.array_equal(msf.read(ours)[0], sf.read(theirs)[0])


def test_mojo_parallel_threshold_with_tail():
    count = 16_777_216 + 3
    values = np.zeros(count, dtype=np.float64)
    encoded = encode_f64(values, 3, 3)
    assert encoded.dtype == np.uint8
    assert encoded.size == count * 3
    assert np.count_nonzero(encoded) == 0
    del values, encoded

    raw = np.zeros(count * 4, dtype=np.uint8)
    decoded = decode_f64(raw, count, 6)
    assert decoded.dtype == np.float64
    assert decoded.size == count
    assert np.count_nonzero(decoded) == 0


def test_mojo_ffi_rejects_invalid_lengths_kinds_and_widths():
    with pytest.raises(ValueError, match="too short"):
        decode_f64(b"\0" * 3, 2, 2)
    with pytest.raises(ValueError, match="non-negative"):
        decode_f64(b"", -1, 1)
    with pytest.raises(ValueError, match="sample kind"):
        decode_f64(b"", 0, 99)
    with pytest.raises(ValueError, match="requires width"):
        encode_f64(np.zeros(2), 3, 4)


def test_read_out_rejects_shapes_that_could_overflow_ffi_buffer(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, signal(20, 2), 48000)
    with pytest.raises(ValueError, match="shape"):
        msf.read(path, out=np.empty(20))
    with msf.SoundFile(path) as stream:
        with pytest.raises(ValueError, match="shape"):
            stream.read(out=np.empty(20))
    readonly = np.empty((20, 2))
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        msf.read(path, out=readonly)


def test_file_descriptor_input_with_caller_owned_lifetime(tmp_path):
    path = tmp_path / "descriptor.wav"
    sf.write(path, signal(32), 8000)
    fd = os.open(path, os.O_RDONLY)
    try:
        actual, rate = msf.read(fd, closefd=False)
        assert rate == 8000
        assert np.array_equal(actual, sf.read(path)[0])
        os.fstat(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("subtype", ["PCM_S8", "PCM_16", "PCM_24"])
def test_flac_roundtrip_matches_upstream(tmp_path, subtype):
    data = signal(6000, 2)
    path = tmp_path / f"audio-{subtype}.flac"
    msf.write(path, data, 44100, subtype=subtype)
    expected, expected_rate = sf.read(path, always_2d=True)
    actual, actual_rate = msf.read(path, always_2d=True)
    assert actual_rate == expected_rate == 44100
    assert np.array_equal(actual, expected)
    details = msf.info(path)
    assert (details.format, details.subtype, details.channels, details.frames) == (
        "FLAC",
        subtype,
        2,
        6000,
    )


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int16"])
def test_read_dtypes_and_shape_match_upstream(tmp_path, dtype):
    path = tmp_path / "stereo.wav"
    sf.write(path, signal(1000, 2), 32000, subtype="PCM_24")
    actual, arate = msf.read(path, dtype=dtype, always_2d=True)
    expected, erate = sf.read(path, dtype=dtype, always_2d=True)
    assert arate == erate
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape == (1000, 2)
    assert np.array_equal(actual, expected)


def test_read_start_stop_frames_and_negative_indices(tmp_path):
    path = tmp_path / "slice.wav"
    sf.write(path, signal(1000), 48000, subtype="PCM_16")
    for kwargs in (
        {"start": 17, "frames": 101},
        {"start": 17, "stop": 211},
        {"start": -100, "stop": -7},
        {"start": 500, "stop": 400},
    ):
        actual, _ = msf.read(path, **kwargs)
        expected, _ = sf.read(path, **kwargs)
        assert np.array_equal(actual, expected)


def test_read_fill_value_and_out_match_upstream(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, signal(20), 48000)
    ours = np.empty(32)
    theirs = np.empty(32)
    actual, _ = msf.read(path, frames=32, fill_value=0.25, out=ours)
    expected, _ = sf.read(path, frames=32, fill_value=0.25, out=theirs)
    assert actual is not None and np.array_equal(actual, expected)
    assert np.array_equal(ours, theirs)


def test_integer_input_write_matches_upstream(tmp_path):
    data = rng.integers(-32768, 32768, size=(2000, 2), dtype=np.int16)
    ours = tmp_path / "ours.wav"
    theirs = tmp_path / "theirs.wav"
    msf.write(ours, data, 22050, subtype="PCM_24")
    sf.write(theirs, data, 22050, subtype="PCM_24")
    assert np.array_equal(sf.read(ours, dtype="int32")[0], sf.read(theirs, dtype="int32")[0])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint64, np.float16, np.complex128])
def test_write_rejects_unsupported_dtype_without_silent_narrowing(tmp_path, dtype):
    with pytest.raises(ValueError, match="dtype must be one of"):
        msf.write(tmp_path / "bad.wav", np.zeros(5, dtype=dtype), 8000)


def test_stateful_write_normalizes_non_native_byte_order(tmp_path):
    path = tmp_path / "endian.wav"
    data = np.array([1, 256, -2], dtype=">i2")
    with msf.SoundFile(path, "w", 8000, 1) as stream:
        stream.write(data)
    actual, _ = msf.read(path, dtype="int16")
    assert np.array_equal(actual, data.astype(np.int16))


def test_soundfile_stateful_read_seek_properties(tmp_path):
    path = tmp_path / "state.flac"
    sf.write(path, signal(500, 2), 16000, subtype="PCM_16")
    with msf.SoundFile(path) as stream:
        assert (stream.samplerate, stream.channels, stream.frames) == (16000, 2, 500)
        assert (stream.format, stream.subtype, stream.endian) == ("FLAC", "PCM_16", "FILE")
        first = stream.read(20, always_2d=True)
        assert stream.tell() == 20
        assert stream.seek(-5, msf.SEEK_CUR) == 15
        second = stream.read(5, always_2d=True)
        assert np.array_equal(second, first[15:20])
        assert len(stream) == 500 and stream.seekable()
    assert stream.closed
    with pytest.raises(RuntimeError):
        stream.tell()


def test_soundfile_incremental_write_and_buffer_io(tmp_path):
    path = tmp_path / "incremental.wav"
    data = signal(101, 2).astype(np.float32)
    with msf.SoundFile(path, "w", 48000, 2, subtype="FLOAT") as stream:
        assert stream.write(data[:40]) == 40
        assert stream.buffer_write(data[40:].tobytes(), "float32") == 61
        assert stream.frames == 101
    with msf.SoundFile(path) as stream:
        raw = stream.buffer_read(dtype="float32")
    assert np.array_equal(np.frombuffer(raw, np.float32).reshape(-1, 2), data)


def test_bytesio_wav_and_flac(tmp_path):
    data = signal(257)
    for fmt in ("WAV", "FLAC"):
        buffer = io.BytesIO()
        msf.write(buffer, data, 12345, format=fmt, subtype="PCM_16")
        buffer.seek(0)
        actual, rate = msf.read(buffer)
        buffer.seek(0)
        expected, expected_rate = sf.read(buffer)
        assert rate == expected_rate == 12345
        assert np.array_equal(actual, expected)


def test_blocks_match_upstream_with_overlap(tmp_path):
    path = tmp_path / "blocks.flac"
    sf.write(path, signal(333), 48000, subtype="PCM_24")
    actual = list(msf.blocks(path, blocksize=64, overlap=11, start=7, stop=301))
    expected = list(sf.blocks(path, blocksize=64, overlap=11, start=7, stop=301))
    assert len(actual) == len(expected)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


def test_metadata_roundtrip(tmp_path):
    path = tmp_path / "metadata.flac"
    with msf.SoundFile(path, "w", 8000, 1, format="FLAC") as stream:
        stream.title = "Mojo tone"
        stream.artist = "Test"
        stream.write(np.zeros(10))
    with msf.SoundFile(path) as stream:
        assert stream.title == "Mojo tone"
        assert stream.artist == "Test"


def test_info_matches_upstream_fields(tmp_path):
    path = tmp_path / "details.wav"
    sf.write(path, signal(250, 2), 25000, subtype="DOUBLE")
    actual, expected = msf.info(path), sf.info(path)
    for name in ("samplerate", "channels", "frames", "format", "subtype", "endian", "sections"):
        assert getattr(actual, name) == getattr(expected, name)
    assert actual.duration == pytest.approx(expected.duration)


def test_argument_errors(tmp_path):
    with pytest.raises(TypeError):
        msf.write(tmp_path / "no-extension", np.zeros(5), 8000)
    with pytest.raises(ValueError):
        msf.write(tmp_path / "bad.flac", np.zeros(5), 8000, subtype="FLOAT")
    with pytest.raises(TypeError):
        list(msf.blocks(tmp_path / "missing.wav"))
