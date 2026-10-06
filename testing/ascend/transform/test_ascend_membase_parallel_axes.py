"""Layout inference retains the logical axes consumed by 2201 MemBase lowering."""

import math

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, nodes


def _dense_program(shape):
    if len(shape) == 2:
        rows, columns = shape

        @T.prim_func
        def main(A: T.Tensor(shape, "float32"), C: T.Tensor(shape, "float32")):
            with T.Kernel(1):
                a = T.alloc_shared(shape, "float32")
                c = T.alloc_shared(shape, "float32")
                T.copy(A, a)
                for i, j in T.Parallel(rows, columns):
                    c[i, j] = a[i, j] + 1.0
                T.copy(c, C)

    else:
        batches, rows, columns = shape

        @T.prim_func
        def main(A: T.Tensor(shape, "float32"), C: T.Tensor(shape, "float32")):
            with T.Kernel(1):
                a = T.alloc_shared(shape, "float32")
                c = T.alloc_shared(shape, "float32")
                T.copy(A, a)
                for b, i, j in T.Parallel(batches, rows, columns):
                    c[b, i, j] = a[b, i, j] + 1.0
                T.copy(c, C)

    return main


def _layout(program, arch="dav-2201"):
    # Use the function's bound target rather than ambient target state. Layout
    # inference and its ordinary nest-layout validator both run unchanged.
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    mod = tirx.transform.BindTarget(target)(tvm.IRModule.from_expr(program))
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)
    mod = tirx.transform.Simplify()(mod)
    with target:
        return transform.AscendLayoutInference()(mod)


@pytest.mark.parametrize("shape", [(16, 64), (2, 8, 64), (24, 64)])
@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_layout_infers_and_validates_the_target_execution_domain(shape, arch):
    after = _layout(_dense_program(shape), arch)
    loops = [loop for loop in nodes(after, tirx.For) if loop.kind == tirx.ForKind.PARALLEL]
    # The established 3510 power-of-two fusion path is unchanged. Its ragged
    # nest, and every 2201 MemBase nest, retain one coordinate per logical axis.
    all_power_of_two = all(extent & (extent - 1) == 0 for extent in shape)
    expected = (math.prod(shape),) if arch == "dav-3510" and all_power_of_two else shape
    assert len(loops) == len(expected)
    outer = next(loop for loop in loops if "parallel_loop_layout" in loop.annotations)
    chain = []
    body = outer
    while isinstance(body, tirx.For) and body.kind == tirx.ForKind.PARALLEL:
        chain.append(body)
        body = body.body
    assert tuple(int(loop.extent) for loop in chain) == expected
    layout = outer.annotations["parallel_loop_layout"]
    assert tuple(int(extent) for extent in layout.get_input_shape()) == expected
    assert all("parallel_loop_layout" not in loop.annotations for loop in chain[1:])
    assert len(nodes(body, tirx.BufferStore)) == 1


def _softmax_step(output_dtype):
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float32"), Row: T.Tensor((16,), "float32"), P: T.Tensor((16, 64), output_dtype)):
        with T.Kernel(1):
            scores = T.alloc_shared((16, 64), "float32")
            row = T.alloc_shared((16,), "float32")
            probability = T.alloc_shared((16, 64), output_dtype)
            T.copy(A, scores)
            T.copy(Row, row)
            for i, j in T.Parallel(16, 64):
                scores[i, j] = T.exp(scores[i, j] - row[i])
                probability[i, j] = T.cast(scores[i, j], output_dtype)
            T.copy(probability, P)

    return main


@pytest.mark.parametrize("output_dtype", ["float16", "bfloat16", "float32"])
def test_layout_then_membase_consumes_a_two_assignment_row_domain(output_dtype):
    laid_out = _layout(_softmax_step(output_dtype))
    loops = nodes(laid_out, tirx.For)
    assert len(loops) == 2
    inner = next(loop for loop in loops if isinstance(loop.body, tirx.SeqStmt))
    assert len(inner.body.seq) == 2
    after = transform.AscendLowerVectorTiles()(laid_out)
    (broadcast,) = calls(after, "tl.ascend_membase.brcb_f32")
    (subtract,) = calls(after, "tl.ascend_membase.sub_rows_f32")
    (exponential,) = calls(after, "tl.ascend_membase.exp")
    assert int(broadcast.args[2]) == 2
    assert int(subtract.args[3]) == 16
    assert int(exponential.args[-1]) == 1024
    assert not nodes(after, tirx.For) and not nodes(after, tirx.Bind)
    if output_dtype != "float32":
        narrow = "float2half_rn" if output_dtype == "float16" else "float2bfloat16_rn"
        (cast,) = calls(after, "tl.ascend_membase." + narrow)
        assert int(cast.args[-1]) == 1024
        source = cast.args[1].args[0].buffer
        destination = exponential.args[0].args[0].buffer
        assert source.data.same_as(destination.data)


def test_preserved_axes_do_not_make_a_strided_ub_access_legal():
    @T.prim_func
    def main(A: T.Tensor((16, 128), "float32"), C: T.Tensor((16, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((16, 128), "float32")
            c = T.alloc_shared((16, 64), "float32")
            T.copy(A, a)
            for i, j in T.Parallel(16, 64):
                c[i, j] = a[i, j * 2]
            T.copy(c, C)

    laid_out = _layout(main)
    assert len(nodes(laid_out, tirx.For)) == 2
    with pytest.raises(ValueError, match="innermost Parallel axis must be contiguous"):
        transform.AscendLowerVectorTiles()(laid_out)
