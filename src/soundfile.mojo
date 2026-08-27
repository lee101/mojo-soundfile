"""PCM conversion kernels exported through a small C ABI."""

from max.algorithm import parallelize
from std.math import round
from std.memory import bitcast
from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime DPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_THRESHOLD = 16777216
comptime GRAIN_SIZE = 262144


def clamp_sample(x: Float64) -> Float64:
    if x != x:
        return -1.0
    return max(-1.0, min(x, 1.0))


@always_inline
def quantize(src: DPtr, i: Int) -> SIMD[DType.int32, W]:
    var values = src.load[width=W](i)
    var lower = SIMD[DType.float64, W](-1.0)
    var upper = SIMD[DType.float64, W](0.9999999995343387)
    values = values.ne(values).select(lower, values)
    values = max(lower, min(values, upper))
    return round(values * 2147483648.0).cast[DType.int32]()


def decode_range(src: BPtr, dst: DPtr, begin: Int, end: Int, kind: Int):
    var i = begin
    if kind == 1:
        while i + W <= end:
            var values = src.load[width=W](i).cast[DType.float64]()
            dst.store(i, (values - 128.0) / 128.0)
            i += W
    elif kind == 2:
        var values = src.bitcast[Int16]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]() / 32768.0,
            )
            i += W
    elif kind == 3:
        while i + 2 * W <= end:
            var j = i * 3
            comptime if W == 4:
                var octets = src.load[width=16](j)
                var expanded = octets.shuffle[
                    0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 0
                ]()
                var values = bitcast[DType.int32, W](expanded)
                values = (values << 8) >> 8
                dst.store(i, values.cast[DType.float64]() / 8388608.0)
            elif W == 8:
                var octets = src.load[width=32](j)
                var expanded = octets.shuffle[
                    0,
                    1,
                    2,
                    0,
                    3,
                    4,
                    5,
                    0,
                    6,
                    7,
                    8,
                    0,
                    9,
                    10,
                    11,
                    0,
                    12,
                    13,
                    14,
                    0,
                    15,
                    16,
                    17,
                    0,
                    18,
                    19,
                    20,
                    0,
                    21,
                    22,
                    23,
                    0,
                ]()
                var values = bitcast[DType.int32, W](expanded)
                values = (values << 8) >> 8
                dst.store(i, values.cast[DType.float64]() / 8388608.0)
            else:
                var low = (src + j).strided_load[width=W](3).cast[DType.int32]()
                var middle = (src + j + 1).strided_load[width=W](3).cast[DType.int32]()
                var high = (src + j + 2).strided_load[width=W](3).cast[DType.int32]()
                var values = ((low | (middle << 8) | (high << 16)) << 8) >> 8
                dst.store(i, values.cast[DType.float64]() / 8388608.0)
            i += W
    elif kind == 4:
        var values = src.bitcast[Int32]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                / 2147483648.0,
            )
            i += W
    elif kind == 6:
        var values = src.bitcast[Float32]()
        while i + W <= end:
            dst.store(
                i, values.load[width=W, alignment=1](i).cast[DType.float64]()
            )
            i += W
    elif kind == 7:
        var values = src.bitcast[Float64]()
        while i + W <= end:
            dst.store(i, values.load[width=W, alignment=1](i))
            i += W
    while i < end:
        if kind == 1:
            dst[i] = (Float64(Int(src[i])) - 128.0) / 128.0
        elif kind == 2:
            var u = Int((src + i * 2).bitcast[UInt16]().load[alignment=1]())
            if u >= 32768:
                u -= 65536
            dst[i] = Float64(u) / 32768.0
        elif kind == 3:
            var j = i * 3
            var u = Int(src[j]) | (Int(src[j + 1]) << 8) | (Int(src[j + 2]) << 16)
            if u >= 8388608:
                u -= 16777216
            dst[i] = Float64(u) / 8388608.0
        elif kind == 4:
            var u = Int64((src + i * 4).bitcast[UInt32]().load[alignment=1]())
            if u >= 2147483648:
                u -= 4294967296
            dst[i] = Float64(u) / 2147483648.0
        elif kind == 6:
            dst[i] = Float64((src + i * 4).bitcast[Float32]().load[alignment=1]())
        else:
            dst[i] = (src + i * 8).bitcast[Float64]().load[alignment=1]()
        i += 1


@export("msf_decode_f64")
def msf_decode_f64(src_addr: Int, dst_addr: Int, n: Int, kind: Int) abi("C"):
    if n <= 0 or src_addr == 0 or dst_addr == 0:
        return
    if kind != 1 and kind != 2 and kind != 3 and kind != 4 and kind != 6 and kind != 7:
        return
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = DPtr(unsafe_from_address=dst_addr)
    if n >= PARALLEL_THRESHOLD:
        var chunks = (n + GRAIN_SIZE - 1) // GRAIN_SIZE

        @parameter
        @__copy_capture(src, dst, n, kind)
        def work(chunk: Int):
            var begin = chunk * GRAIN_SIZE
            decode_range(src, dst, begin, min(begin + GRAIN_SIZE, n), kind)

        parallelize[work](chunks, min(chunks, 8))
    else:
        decode_range(src, dst, 0, n, kind)


def encode_range(src: DPtr, dst: BPtr, begin: Int, end: Int, kind: Int):
    var i = begin
    if kind == 1:
        var values = dst.bitcast[UInt8]()
        while i + W <= end:
            var q = ((quantize(src, i) >> 24) + 128).cast[DType.uint8]()
            values.store[alignment=1](i, q)
            i += W
        while i < end:
            var q32 = Int64(round(clamp_sample(src[i]) * 2147483648.0))
            q32 = max(Int64(-2147483648), min(q32, Int64(2147483647)))
            var q = Int(q32 >> 24) + 128
            dst[i] = UInt8(q)
            i += 1
    elif kind == 2:
        var values = dst.bitcast[Int16]()
        while i + W <= end:
            var q = (quantize(src, i) >> 16).cast[DType.int16]()
            values.store[alignment=1](i, q)
            i += W
        while i < end:
            var q32 = Int64(round(clamp_sample(src[i]) * 2147483648.0))
            q32 = max(Int64(-2147483648), min(q32, Int64(2147483647)))
            var q = Int(q32 >> 16)
            var u = q if q >= 0 else q + 65536
            dst[i * 2] = UInt8(u & 255)
            dst[i * 2 + 1] = UInt8((u >> 8) & 255)
            i += 1
    elif kind == 3:
        while i + W <= end:
            var packed = quantize(src, i) >> 8
            comptime if W == 4:
                var octets = bitcast[DType.uint8, 16](packed)
                var compact = octets.shuffle[
                    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0
                ]()
                dst.store(i * 3, compact.slice[8]())
                dst.store(i * 3 + 8, compact.slice[4, offset=8]())
            elif W == 8:
                var octets = bitcast[DType.uint8, 32](packed)
                var compact = octets.shuffle[
                    0,
                    1,
                    2,
                    4,
                    5,
                    6,
                    8,
                    9,
                    10,
                    12,
                    13,
                    14,
                    16,
                    17,
                    18,
                    20,
                    21,
                    22,
                    24,
                    25,
                    26,
                    28,
                    29,
                    30,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]()
                dst.store(i * 3, compact.slice[16]())
                dst.store(i * 3 + 16, compact.slice[8, offset=16]())
            else:
                for lane in range(W):
                    var q = Int(packed[lane])
                    var u = q if q >= 0 else q + 16777216
                    var j = (i + lane) * 3
                    dst[j] = UInt8(u & 255)
                    dst[j + 1] = UInt8((u >> 8) & 255)
                    dst[j + 2] = UInt8((u >> 16) & 255)
            i += W
        while i < end:
            var q32 = Int64(round(clamp_sample(src[i]) * 2147483648.0))
            q32 = max(Int64(-2147483648), min(q32, Int64(2147483647)))
            var q = Int(q32 >> 8)
            var u = q if q >= 0 else q + 16777216
            var j = i * 3
            dst[j] = UInt8(u & 255)
            dst[j + 1] = UInt8((u >> 8) & 255)
            dst[j + 2] = UInt8((u >> 16) & 255)
            i += 1
    elif kind == 4:
        var values = dst.bitcast[Int32]()
        while i + W <= end:
            var q = quantize(src, i).cast[DType.int32]()
            values.store[alignment=1](i, q)
            i += W
        while i < end:
            var q32 = Int64(round(clamp_sample(src[i]) * 2147483648.0))
            q32 = max(Int64(-2147483648), min(q32, Int64(2147483647)))
            var q = q32
            var u = q if q >= 0 else q + 4294967296
            var j = i * 4
            dst[j] = UInt8(u & 255)
            dst[j + 1] = UInt8((u >> 8) & 255)
            dst[j + 2] = UInt8((u >> 16) & 255)
            dst[j + 3] = UInt8((u >> 24) & 255)
            i += 1
    elif kind == 6:
        var values = dst.bitcast[Float32]()
        while i + W <= end:
            values.store[alignment=1](
                i, src.load[width=W](i).cast[DType.float32]()
            )
            i += W
        while i < end:
            (dst + i * 4).bitcast[Float32]().store[alignment=1](0, Float32(src[i]))
            i += 1
    else:
        var values = dst.bitcast[Float64]()
        while i + W <= end:
            values.store[alignment=1](i, src.load[width=W](i))
            i += W
        while i < end:
            (dst + i * 8).bitcast[Float64]().store[alignment=1](0, src[i])
            i += 1


@export("msf_encode_f64")
def msf_encode_f64(src_addr: Int, dst_addr: Int, n: Int, kind: Int) abi("C"):
    if n <= 0 or src_addr == 0 or dst_addr == 0:
        return
    if kind != 1 and kind != 2 and kind != 3 and kind != 4 and kind != 6 and kind != 7:
        return
    var src = DPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    if n >= PARALLEL_THRESHOLD:
        var chunks = (n + GRAIN_SIZE - 1) // GRAIN_SIZE

        @parameter
        @__copy_capture(src, dst, n, kind)
        def work(chunk: Int):
            var begin = chunk * GRAIN_SIZE
            encode_range(src, dst, begin, min(begin + GRAIN_SIZE, n), kind)

        parallelize[work](chunks, min(chunks, 8))
    else:
        encode_range(src, dst, 0, n, kind)
