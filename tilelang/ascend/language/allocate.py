"""Ascend on-chip buffer allocation: L1 (CBuf) and the L0A/L0B/L0C Cube scopes."""

from __future__ import annotations

from contextlib import contextmanager

from tvm import arith

from tilelang._typing import DType, ShapeType
from tilelang.language.allocate import _with_span
from tvm import DataType
from tvm.script import tirx as T
from tvm.tirx.buffer import Buffer
from tvm.tirx.script.builder.ir import sblock_attr


@contextmanager
def AIVPrivate():
    """Declare UB allocations in this scope as independently owned by each AIV.

    Use once around the complete private UB declarations in the kernel root.
    The scope emits no block, allocation or runtime operation of its own.
    Conditional Python allocations are included only when created. All existing
    ``annotate_aiv_private`` ownership and disjoint-write checks still apply.
    Do not nest scopes or combine with an explicit private declaration in the
    same block. Allocate only UB buffers here; L1/L0/workspace stay outside.
    """
    from tvm.script.ir_builder import IRBuilder
    from tvm.tirx.script.builder.frame import SBlockFrame
    from .annotations import annotate_aiv_private

    frames = IRBuilder.current().frames
    blocks = [frame for frame in frames if isinstance(frame, SBlockFrame)]
    if len(blocks) != 1:
        raise ValueError("AIVPrivate requires the kernel root block")
    root = blocks[0]
    if not frames[-1].same_as(root):
        raise ValueError("AIVPrivate declarations must be outside control flow")
    if root.annotations is not None and "tl.ascend_aiv_private" in root.annotations:
        raise ValueError("AIVPrivate requires one private declaration per block")
    begin = len(root.alloc_buffers)
    yield
    if root.annotations is not None and "tl.ascend_aiv_private" in root.annotations:
        raise ValueError("AIVPrivate scopes cannot nest or contain another private declaration")
    annotate_aiv_private(*list(root.alloc_buffers)[begin:])


def alloc_shared(shape: ShapeType, dtype: DType, scope="shared.dyn") -> Buffer:
    """Allocate a UB buffer.

    Same surface as the common ``T.alloc_shared`` minus its CUDA-only bool
    workaround (bool buffers there downgrade to the static "shared" scope
    because the smem-merge pass cannot handle bool): Ascend UB handles bool
    buffers in the requested scope directly.
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_workspace(shape: ShapeType, dtype: DType) -> Buffer:
    """Allocate private runtime-managed GM storage eligible for versioning.

    The shape describes one logical generation, including any grid/owner axes.
    The ordinary buffer owner and dependency analysis determine its lifetime;
    stages and version hints determine bounded physical storage. Host lowering
    allocates aligned storage on the caller's stream and owns its lifetime.
    This declaration cannot register an external tensor or enlarge its ABI.
    Allocate it in the kernel's root block, as with on-chip tiles.
    """
    buffer = _with_span(T.sblock_alloc_buffer(shape, dtype, scope="global"))
    # Maps merge across allocator calls in the script builder; distinct array
    # annotations would conflict rather than append their storage identities.
    sblock_attr({"tl.ascend.compiler_workspace_storages": {buffer.data: T.int32(1)}})
    return buffer


def alloc_l1(shape: ShapeType, dtype: DType, scope="shared.l1") -> Buffer:
    """Allocate an L1 buffer (__cbuf__) on Ascend NPU.

    Args:
        shape: Buffer shape
        dtype: Data type
        scope: Memory scope. Defaults to "shared.l1"
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0a(shape: ShapeType, dtype: DType, scope="shared.l0a") -> Buffer:
    """Allocate an L0A buffer (__ca__) on Ascend NPU.

    Layout is deferred to the consuming gemm operation, which infers K-major
    or MN-major from its transpose flags.
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0b(shape: ShapeType, dtype: DType, scope="shared.l0b") -> Buffer:
    """Allocate an L0B buffer (__cb__) on Ascend NPU.

    Layout is deferred to the consuming gemm operation, which infers K-major
    or MN-major from its transpose flags.
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0c(shape: ShapeType, dtype: DType, scope="shared.l0c", layout: bool = True) -> Buffer:
    """Allocate an L0C buffer (__cc__) on Ascend NPU.

    Args:
        shape: Buffer shape
        dtype: Data type
        scope: Memory scope. Defaults to "shared.l0c"
        layout: Whether to annotate the fixed Ascend L0C accumulator layout.
    """
    buf = _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))
    if layout:
        from tilelang.layout import make_ascend_l0c_layout

        if len(buf.shape) >= 2:
            sblock_attr({"layout_map": {buf.data: make_ascend_l0c_layout(buf)}})
    return buf


# One E8M0 scale factor covers 32 K elements; wider SF dtypes pack several
# scales per element (uint16 = one pair per 64 K elements).
_MX_SF_K_PER_SCALE_BYTE = 32


def _alloc_l0_sf(buf: Buffer, expected_scope: str, sf_dtype: DType, sf_shape: ShapeType | None) -> Buffer:
    if buf.scope() != expected_scope:
        raise ValueError(f"expected a {expected_scope} data tile, got scope {buf.scope()} for {buf.name}")
    if len(buf.shape) < 2:
        raise ValueError(f"L0 SF requires trailing matrix dimensions, got {buf.shape}")
    dtype = DataType(sf_dtype)
    if dtype.bits not in (8, 16) or dtype.lanes != 1:
        raise ValueError(f"L0 SF storage must contain 8-bit scales or 16-bit scale pairs, got {sf_dtype}")
    if sf_shape is None:
        k_per_sf = _MX_SF_K_PER_SCALE_BYTE * (dtype.bits // 8)
        rows, k = buf.shape[-2], buf.shape[-1]
        if (isinstance(k, int) or hasattr(k, "value")) and int(k) % k_per_sf != 0:
            raise ValueError(
                f"data tile K extent {k} of {buf.name} is not divisible by {k_per_sf} "
                f"(K elements per {sf_dtype} scale element); pass sf_shape explicitly for a transposed tile"
            )
        sf_shape = (*buf.shape[:-2], rows, k // k_per_sf)
    analyzer = arith.Analyzer()
    if len(sf_shape) != len(buf.shape) or any(not analyzer.can_prove_equal(a, b) for a, b in zip(sf_shape[:-2], buf.shape[:-2])):
        raise ValueError(f"L0 SF shape {sf_shape} must preserve the data buffer's leading dimensions {buf.shape[:-2]}")
    sf = _with_span(T.sblock_alloc_buffer(sf_shape, sf_dtype, scope=expected_scope + ".sf"))
    sblock_attr({"tl.l0_sf_bindings": {sf.data: buf.data}})
    return sf


def alloc_l0a_sf(buf: Buffer, sf_dtype: DType = "uint16", sf_shape: ShapeType | None = None) -> Buffer:
    """Return the MX scale-factor handle of an L0A data tile.

    The handle (scope ``shared.l0a.sf``) materializes no storage: the
    hardware keys a tile's scale slots to the tile's own address. Load
    scales with ``T.copy(sf_l1, handle)`` and pass the handle to
    ``T.gemm_blockscaled`` as ``SFA``. Allocation permanently binds this
    handle to ``buf``; GEMM verifies the binding and leading indices. Scale
    slots are sticky, so one scale load may serve several data loads (see
    testing/ascend/language/test_tilelang_ascend_mx_sf_slots.py).

    The default shape preserves all leading dimensions and assumes trailing
    untransposed ``(rows, K)`` matrix dimensions with
    one scale per 32 K elements packed into ``sf_dtype`` (``uint16`` = one
    pair per 64 K elements); pass ``sf_shape`` explicitly for transposed
    tiles. Allocate one SF handle per data buffer and reuse that handle.
    """
    return _alloc_l0_sf(buf, "shared.l0a", sf_dtype, sf_shape)


def alloc_l0b_sf(buf: Buffer, sf_dtype: DType = "uint16", sf_shape: ShapeType | None = None) -> Buffer:
    """Return the MX scale-factor handle of an L0B data tile.

    See :func:`alloc_l0a_sf`; identical semantics for the L0B slot shadow.
    """
    return _alloc_l0_sf(buf, "shared.l0b", sf_dtype, sf_shape)


__all__ = [
    "AIVPrivate",
    "alloc_shared",
    "alloc_workspace",
    "alloc_l1",
    "alloc_l0a",
    "alloc_l0a_sf",
    "alloc_l0b",
    "alloc_l0b_sf",
    "alloc_l0c",
]
