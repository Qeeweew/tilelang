"""Ascend dialect of the loop constructs: the common ops plus Ascend knobs."""

from __future__ import annotations

from typing import Any

from tvm import tirx
from tvm.tirx.script.builder import frame

from tilelang.language.loop import Pipelined as _common_pipelined
from tilelang.language.loop import unroll as _common_unroll
from tilelang.language.loop import serial as _serial

__all__ = ["Pipelined", "ManualPipeline", "unroll"]


def ManualPipeline(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    *,
    preserve_issue_order: bool = True,
) -> frame.ForFrame:
    """Construct a logical task loop whose offsets are specified by ``Stage``.

    The default preserves complete source issue order. With
    ``preserve_issue_order=False``, stages and per-pipe source order remain
    fixed while the scheduler may choose cross-pipe order. Buffer versions,
    fill/drain and synchronization retain the ordinary compiler contracts.
    This is a frontend spelling of the existing serial-loop annotations;
    it introduces neither automatic pipeline intent nor a version limit.
    """
    if not isinstance(preserve_issue_order, bool):
        raise ValueError("ManualPipeline preserve_issue_order must be a bool")
    return _serial(
        start,
        stop,
        annotations={"enable_offset": True, "tl.ascend_preserve_issue_order": preserve_issue_order},
    )


def Pipelined(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    num_stages: int | None = None,
    order: list[int] | None = None,
    stage: list[int] | None = None,
    annotations: dict[str, Any] | None = None,
    *,
    max_stages: int = 4,
) -> frame.ForFrame:
    """Construct an Ascend loop with automatic pipeline selection by default.

    With ``num_stages=None``, the native scheduler jointly chooses issue order,
    task stages and physical buffer versions under target storage constraints.
    ``max_stages`` bounds the search; its default of four is an optimization
    heuristic, not a requested stage count or a measured latency prediction.
    The backend further bounds the search by the active hardware pipe chain.
    This control can be varied to diagnose the benefit and cost of deeper
    pipelines without changing the kernel's algorithm.

    An explicit positive ``num_stages`` preserves the upstream meaning: the
    maximum number of physical buffer versions, with ``enable_offset`` in
    ``annotations`` controlling task stage offsets. An explicit zero retains
    the serial default and disables automatic pipeline selection. Manual
    ``T.Stage`` scopes retain their requested stages.

    The common TileLang dialect and its default ``num_stages=0`` are unchanged.
    """
    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if num_stages is None:
        if not isinstance(max_stages, int) or isinstance(max_stages, bool) or max_stages < 1:
            raise ValueError(f"Ascend Pipelined max_stages must be a positive integer, got {max_stages!r}")
        ann["tl.ascend_auto_pipeline"] = True
        # Keep the bound in upstream's durable num_stages annotation. Control
        # normalization also needs it when choosing a synthetic while epoch.
        num_stages = max_stages
    elif not isinstance(num_stages, int) or isinstance(num_stages, bool) or num_stages < 0:
        raise ValueError(f"Ascend Pipelined num_stages must be None or a non-negative integer, got {num_stages!r}")
    else:
        # Preserve an explicit opt-out through lowering, including zero whose
        # common loop builder omits num_stages. Nested native GEMM loops must
        # not inherit a surrounding automatic request across this boundary.
        ann["tl.ascend_auto_pipeline"] = False
    return _common_pipelined(start, stop, num_stages, order, stage, ann)


def unroll(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    step: tirx.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """The unrolled For statement, with the Ascend unroll-factor pragma.

    Same semantics as the common :func:`tilelang.language.loop.unroll`.
    ``unroll_factor`` is lowered as the ``"pragma_unroll_factor"`` annotation,
    which the Ascend codegen emits as ``#pragma unroll N``.

    Parameters
    ----------
    start, stop, step : PrimExpr
        Iteration range.
    explicit : bool
        Whether to explicitly unroll the loop at compile time.
    unroll_factor : Optional[int]
        Partial unroll factor, lowered as the ``"pragma_unroll_factor"``
        annotation.
    annotations : Optional[Dict[str, Any]]
        Additional loop annotations; values in it take precedence.
    """
    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if unroll_factor is not None:
        ann.setdefault("pragma_unroll_factor", unroll_factor)
    return _common_unroll(start, stop, step, explicit=explicit, annotations=ann or None)
