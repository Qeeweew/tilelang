"""Coalesce independent owned rows without changing serial dependencies."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform

from testing.ascend._ir import calls, nodes


def _private_rows(program, *, partitioned=True):
    mod = tirx.transform.BindTarget(tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))(
        tvm.IRModule.from_expr(program)
    )
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True, lower_thread_binding=False, default_threads=None,
        unsupported_annotations=["cluster_dims"], launch_dim_tags=["cthread"],
    )(mod)
    mod = tirx.transform.Simplify()(mod)

    def own_rows(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL and isinstance(node.body, tirx.For):
            return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.SERIAL, node.body,
                            annotations=node.annotations)
        if partitioned and isinstance(node, tirx.SBlock) and node.name_hint == "tilelang_root":
            return tirx.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint, node.body,
                              init=node.init, alloc_buffers=node.alloc_buffers, match_buffers=node.match_buffers,
                              annotations={**dict(node.annotations), "tl.ascend_row_partitioned": 2})

    return tvm.IRModule({gvar: func.with_body(ir_transform(func.body, None, own_rows))
                        for gvar, func in mod.functions.items()})


def _dense_program(rows, columns, dtype):
    @T.prim_func
    def main(A: T.Tensor((rows, columns), dtype), B: T.Tensor((rows, columns), dtype),
             C: T.Tensor((rows, columns), dtype)):
        with T.Kernel(1):
            matrix = T.alloc_shared((rows, columns), dtype)
            addend = T.alloc_shared((rows, columns), dtype)
            result = T.alloc_shared((rows, columns), dtype)
            T.copy(A, matrix)
            T.copy(B, addend)
            for i, j in T.Parallel(rows, columns):
                matrix[i, j] = matrix[i, j] * T.cast(0.5, dtype)
                result[i, j] = matrix[i, j] + addend[i, j]
            T.copy(result, C)

    return main


@pytest.mark.parametrize("rows,columns", [(3, 80), (9, 128), (1, 32)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_dense_owned_rows_use_whole_tile_count_with_typed_node_boundaries(rows, columns, dtype):
    after = transform.AscendLowerVectorTiles()(_private_rows(_dense_program(rows, columns, dtype)))
    assert not nodes(after, tirx.For) and not nodes(after, tirx.Bind)
    assert not calls(after, "tl.ascend_membase.brcb_f32")
    native = [call for call in nodes(after, tirx.Call)
              if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]
    assert all(int(call.args[-1]) == rows * columns for call in native)
    multiply = "mul" if dtype == "bfloat16" else "mul_scalar"
    assert len(calls(after, "tl.ascend_membase." + multiply)) == 1
    assert len(calls(after, "tl.ascend_membase.add")) == 1
    if dtype == "bfloat16":
        # The written first assignment rounds before the second one reads it.
        assert len(calls(after, "tl.ascend_membase.float2bfloat16_rn")) == 2


def test_unowned_serial_rows_keep_their_serial_control():
    after = transform.AscendLowerVectorTiles()(
        _private_rows(_dense_program(3, 80, "float32"), partitioned=False)
    )
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))
    assert int(calls(after, "tl.ascend_membase.add")[0].args[-1]) == 80


def test_cross_row_alias_keeps_the_serial_recurrence():
    @T.prim_func
    def main(A: T.Tensor((5, 64), "float32"), C: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            matrix = T.alloc_shared((5, 64), "float32")
            T.copy(A, matrix)
            for i, j in T.Parallel(4, 64):
                matrix[i + 1, j] = matrix[i, j] + 1.0
            T.copy(matrix[1:5, :], C)

    after = transform.AscendLowerVectorTiles()(_private_rows(main))
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))
    assert int(calls(after, "tl.ascend_membase.add_scalar")[0].args[-1]) == 64
