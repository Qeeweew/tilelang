"""Ascend dialect of the reduce operators.

Same surface as :mod:`tilelang.language.reduce_op`. Shared-to-shared reductions
retain their UB regions: native MemBase lowering owns the 2201 path, and
AscendSimdVFLowerParallel owns reductions inside a 3510 SimdVF. Neither path
uses the common GPU fragment round-trip.

The thin ``reduce_*`` wrappers are redeclared because the common ones call
the common ``reduce`` by module-level binding and would bypass this dialect's
dispatch.
"""

from __future__ import annotations

from tvm import tirx

from tilelang.language.common import evaluate
from tilelang.language.reduce_op import (
    _REDUCE_OP_KEY,
    _legalize_dim,
    reduce as _common_reduce,
)
from tilelang.language.utils import _normalize_annotations
from tilelang.utils.language import is_shared, retrieve_shape, to_buffer_region, to_tile_region

__all__ = [
    "reduce",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
    "reduce_abssum",
    "reduce_absmax",
    "reduce_bitand",
    "reduce_bitor",
    "reduce_bitxor",
]


def reduce(
    buffer: tirx.Buffer,
    out: tirx.Buffer,
    reduce_type: str,
    dim: int,
    clear: bool,
    batch: int = 1,
    annotations: dict | None = None,
) -> None:
    """Emit UB reduction intent without introducing GPU fragments."""
    if not (is_shared(buffer) and is_shared(out)):
        return _common_reduce(buffer, out, reduce_type, dim, clear, batch=batch, annotations=annotations)

    # Preserve logical regions until the bound target selects an instruction
    # algorithm and exposes its scratch and effects before scheduling.
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    out_buffer = to_buffer_region(out).buffer
    if reduce_type in ("bitand", "bitor", "bitxor") and not (out_buffer.dtype.startswith(("int", "uint")) or out_buffer.dtype == "bool"):
        raise ValueError(f"reduce_{reduce_type} requires an integer/bool buffer, got dtype {out_buffer.dtype}")
    buf_shape = retrieve_shape(buffer)
    out_shape = retrieve_shape(out)
    expected_shapes = [buf_shape[:dim] + buf_shape[dim + 1 :], buf_shape[:dim] + [1] + buf_shape[dim + 1 :]]
    if list(out_shape) not in expected_shapes:
        expected_shapes_str = " or ".join(map(str, expected_shapes))
        raise ValueError(
            f"Invalid reduce output shape, buffer shape is {buf_shape}, dim is {dim}, "
            f"output shape is {out_shape}, expected shapes are {expected_shapes_str}"
        )

    annotations = _normalize_annotations(annotations)
    if batch > 1:
        annotations["batch"] = batch

    return evaluate(
        tirx.call_intrin(
            "handle",
            tirx.op.Op.get(_REDUCE_OP_KEY),
            to_tile_region(buffer, access_type="r"),
            to_tile_region(out, access_type="w" if clear else "rw"),
            reduce_type,
            dim,
            clear,
            annotations=annotations,
        )
    )


def reduce_max(
    buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1, clear: bool = True, batch: int = 1, annotations: dict | None = None
) -> None:
    """See :func:`tilelang.language.reduce_op.reduce_max`."""
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "max", dim, clear, batch=batch, annotations=annotations)


def reduce_min(
    buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1, clear: bool = True, batch: int = 1, annotations: dict | None = None
) -> None:
    """See :func:`tilelang.language.reduce_op.reduce_min`."""
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "min", dim, clear, batch=batch, annotations=annotations)


def reduce_sum(
    buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1, clear: bool = True, batch: int = 1, annotations: dict | None = None
) -> None:
    """See :func:`tilelang.language.reduce_op.reduce_sum`."""
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "sum", dim, clear, batch=batch, annotations=annotations)


def reduce_abssum(buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1, batch: int = 1, annotations: dict | None = None) -> None:
    """See :func:`tilelang.language.reduce_op.reduce_abssum`."""
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "abssum", dim, True, batch=batch, annotations=annotations)


def reduce_absmax(
    buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1, clear: bool = True, batch: int = 1, annotations: dict | None = None
) -> None:
    """See :func:`tilelang.language.reduce_op.reduce_absmax`."""
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "absmax", dim, clear, batch=batch, annotations=annotations)


def reduce_bitand(
    buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1, clear: bool = True, batch: int = 1, annotations: dict | None = None
) -> None:
    """See :func:`tilelang.language.reduce_op.reduce_bitand`."""
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitand", dim, clear, batch=batch, annotations=annotations)


def reduce_bitor(
    buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1, clear: bool = True, batch: int = 1, annotations: dict | None = None
) -> None:
    """See :func:`tilelang.language.reduce_op.reduce_bitor`."""
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitor", dim, clear, batch=batch, annotations=annotations)


def reduce_bitxor(
    buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1, clear: bool = True, batch: int = 1, annotations: dict | None = None
) -> None:
    """See :func:`tilelang.language.reduce_op.reduce_bitxor`."""
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitxor", dim, clear, batch=batch, annotations=annotations)
