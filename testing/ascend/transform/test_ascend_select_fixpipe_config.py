"""Invariant 2201 FIX state has explicit root ownership and completion."""

from collections import Counter, defaultdict, deque
from unittest.mock import patch

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.ascend.transform import z3_scheduler
from tilelang.layout import make_ascend_l0c_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


FLAG = "tl.ascend.root_fixpipe_config"
CONFIG = "tl.ascend_set_fixpipe_config"
COPY = "tl.tileop.ascend_copy"
LATE = "tl.ascend_copy_matrix_cc_to_gm"
TARGET = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})


def _target(mod, arch="dav-2201"):
    return tvm.IRModule({"main": mod["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _fixture(body=None):
    src = [tirx.decl_buffer((16, 16), "float32", name=f"acc{i}", scope="shared.l0c")
           for i in range(2)]
    dst = [tirx.decl_buffer((4, 16, 16), "float32", name=f"out{i}") for i in range(2)]
    i = tirx.Var("i", "int32")
    operations = [copy(a, b, dst_indices=[i, 0, 0], dst_shape=[1, 16, 16],
                       annotations={"unit_flag_ctrl": tirx.IntImm("int32", 0)})
                  for a, b in zip(src, dst)]
    statements = seq(*operations)
    if body:
        statements = body(statements)
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, statements)
    return _target(kernel(loop, buffers=src, params=dst, annotations={
        "layout_map": {buffer: make_ascend_l0c_layout(buffer) for buffer in src},
    }))


def _select(mod):
    with mod["main"].attrs["target"]:
        return transform.AscendSelectFixpipeConfig()(mod)


def test_selection_keeps_copy_operands_and_all_existing_hints():
    before = _fixture()
    after = _select(before)
    assert len(calls(after, CONFIG)) == 1
    for original, selected in zip(calls(before, COPY), calls(after, COPY)):
        tvm.ir.assert_structural_equal(original.args, selected.args)
        for key, value in original.annotations.items():
            tvm.ir.assert_structural_equal(value, selected.annotations[key])
        assert int(selected.annotations[FLAG]) == 1
    with TARGET:
        late = transform.AscendLowerTileOp()(after)
    assert len(calls(late, LATE)) == 2
    assert all(int(call.annotations[FLAG]) == 1 for call in calls(late, LATE))
    for call in calls(late, LATE):
        assert tvm.arith.Analyzer().can_prove_equal(call.args[6], 16)


def test_selection_is_idempotent_and_3510_is_unchanged():
    selected = _select(_fixture())
    tvm.ir.assert_structural_equal(selected, _select(selected))
    baseline = _target(_fixture(), "dav-3510")
    tvm.ir.assert_structural_equal(baseline, _select(baseline))


def test_multiple_kernels_each_own_their_configuration():
    first, second = _fixture(), _fixture()
    parameters = [*first["main"].params, *second["main"].params]
    buffer_map = {**dict(first["main"].buffer_map), **dict(second["main"].buffer_map)}
    function = tirx.PrimFunc(parameters, seq(first["main"].body, second["main"].body),
                             buffer_map=buffer_map).with_attr({
                                 "target": TARGET, "global_symbol": "two_kernels"})
    selected = _select(tvm.IRModule({"main": function}))
    assert len(calls(selected, CONFIG)) == 2
    assert len(calls(selected, COPY)) == 4


@pytest.mark.parametrize("kind", ["extern", "pure_extern", "raw_setter", "late_fix",
                                  "conditional", "task", "per_core"])
def test_unverified_state_or_atomic_consumers_keep_complete_local_path(kind):
    def wrap(body):
        if kind in ("extern", "pure_extern", "raw_setter"):
            symbol = "asc_set_l0c_copy_params" if kind == "raw_setter" else "foreign"
            op = "tirx.call_pure_extern" if kind == "pure_extern" else "tirx.call_extern"
            return seq(tirx.Evaluate(tirx.Call("int32", op, [symbol])), body)
        if kind == "late_fix":
            return seq(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(LATE), [0] * 25)), body)
        if kind == "conditional":
            return tirx.IfThenElse(tirx.Var("condition", "bool"), body, None)
        task = unit(body, core=2).body
        if kind == "task":
            return task
        return tirx.AttrStmt({"core_mask": tirx.IntImm("int64", 2)},
                            "tl.ascend_per_core_task", 1, task)
    before = _fixture(wrap)
    after = _select(before)
    tvm.ir.assert_structural_equal(before, after)
    assert not calls(after, CONFIG)


def test_internal_selected_copy_without_root_is_rejected_by_late_lowering():
    selected = _select(_fixture())
    root = selected["main"].body.block
    root = tirx.SBlock([], [], [], "tilelang_root", root.body.seq[1],
                      alloc_buffers=root.alloc_buffers, annotations=root.annotations)
    selected["main"] = selected["main"].with_body(tirx.SBlockRealize([], True, root))
    with TARGET, pytest.raises(ValueError, match="preceding unconditional kernel-root configuration"):
        transform.AscendLowerTileOp()(selected)


def test_internal_selected_native_copy_without_root_is_rejected_by_codegen():
    call = tirx.Call("void", tvm.ir.Op.get(LATE), [0] * 25, annotations={FLAG: tirx.IntImm("int32", 1)})
    function = tirx.PrimFunc([], tirx.Evaluate(call)).with_attr({
        "target": TARGET, "global_symbol": "unconfigured", "calling_conv": 2,
    })
    build = tvm.ffi.get_global_func("target.build.tilelang_ascend_without_compile")
    with pytest.raises(ValueError, match="preceding unconditional kernel-root configuration"):
        build(tvm.IRModule({"unconfigured": function}), TARGET)


@pytest.mark.parametrize("kind", ["loop", "condition", "vector", "foreign", "unselected"])
def test_forged_root_contract_is_rejected(kind):
    selected = _select(_fixture())
    root = selected["main"].body.block
    setup, body = root.body.seq
    if kind == "loop":
        setup = tirx.For(tirx.Var("repeated", "int32"), 0, 2, tirx.ForKind.SERIAL, setup)
    elif kind == "condition":
        setup = tirx.IfThenElse(tirx.Var("enabled", "bool"), setup, None)
    elif kind == "vector":
        setup = tirx.AttrStmt({"core_mask": tirx.IntImm("int64", 1)},
                             "tl.ascend_task", 1, setup.body)
    elif kind == "foreign":
        body = seq(tirx.Evaluate(tirx.Call("int32", "tirx.call_extern", ["foreign"])), body)
    else:
        original = calls(_fixture(), COPY)[0]
        body = seq(tirx.Evaluate(original), body)
    root = tirx.SBlock([], [], [], "tilelang_root", seq(setup, body),
                      alloc_buffers=root.alloc_buffers, annotations=root.annotations)
    selected["main"] = selected["main"].with_body(tirx.SBlockRealize([], True, root))
    with TARGET, pytest.raises(ValueError, match="root configuration|Root Fixpipe configuration"):
        transform.AscendLowerTileOp()(selected)


def _scheduled_state(*, count=4):
    selected = _select(_fixture())
    root = selected["main"].body.block
    setup = unit(tirx.Evaluate(calls(selected, CONFIG)[0]), core=2, cost=(1, 1))
    original_loop = root.body.seq[1]
    body = seq(*(unit(tirx.Evaluate(call), core=2, cost=(1, 1))
                 for call in calls(selected, COPY)))
    loop = tirx.For(original_loop.loop_var, 0, count, tirx.ForKind.SERIAL, body)
    return _target(kernel(seq(setup, unit(loop, core=None)),
                          buffers=root.alloc_buffers,
                          params=list(selected["main"].buffer_map.values()),
                          annotations=root.annotations))


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("iterations", [0, 1, 4])
def test_root_completion_is_balanced_and_not_consumed_by_each_iteration(dynamic, iterations):
    count = tirx.Var("count", "int32") if dynamic else iterations
    before = _scheduled_state(count=count)
    if dynamic:
        before["main"] = tirx.PrimFunc(
            [*before["main"].params, count], before["main"].body,
            buffer_map=before["main"].buffer_map, attrs=before["main"].attrs)
    with TARGET:
        after = transform.InsertSync()(before)

    # The invariant setup always executes once. A statically nonempty loop
    # may legally absorb its wait under the first-iteration guard; a possibly
    # empty loop must drain that same event outside its iteration domain.
    # Count executions through the actual emitted control instead of requiring
    # a particular textual placement of the wait.
    observed = {"setup": 0, "sets": Counter(), "waits": Counter()}
    analyzer = tvm.arith.Analyzer()

    def scalar(expr, bindings):
        value = analyzer.simplify(tirx.stmt_functor.substitute(expr, bindings))
        assert isinstance(value, tirx.IntImm), (expr, value, after.script())
        return int(value)

    def execute(stmt, bindings):
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                execute(child, bindings)
        elif isinstance(stmt, tirx.SBlockRealize):
            assert not stmt.iter_values
            if scalar(stmt.predicate, bindings):
                execute(stmt.block, bindings)
        elif isinstance(stmt, (tirx.SBlock, tirx.AttrStmt)):
            execute(stmt.body, bindings)
        elif isinstance(stmt, tirx.For):
            start = scalar(stmt.min, bindings)
            extent = scalar(stmt.extent, bindings)
            step = scalar(stmt.step, bindings) if stmt.step is not None else 1
            assert step > 0
            for value in range(start, start + extent, step):
                nested = dict(bindings)
                nested[stmt.loop_var] = tirx.IntImm(stmt.loop_var.dtype, value)
                execute(stmt.body, nested)
        elif isinstance(stmt, tirx.IfThenElse):
            branch = stmt.then_case if scalar(stmt.condition, bindings) else stmt.else_case
            if branch is not None:
                execute(branch, bindings)
        elif isinstance(stmt, tirx.Bind):
            bindings[stmt.var] = tirx.IntImm(stmt.var.dtype, scalar(stmt.value, bindings))
        elif isinstance(stmt, tirx.Evaluate):
            call = stmt.value
            if not isinstance(call, tirx.Call) or not isinstance(call.op, tvm.ir.Op):
                return
            if call.op.name == CONFIG:
                observed["setup"] += 1
            elif call.op.name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
                if call.args[0].value == "S_FIX":
                    kind = "sets" if call.op.name == "tl.ascend_set_flag" else "waits"
                    observed[kind][scalar(call.args[1], bindings)] += 1
        else:
            pytest.fail(f"Unexpected control statement: {stmt}")

    bindings = {count: tirx.IntImm("int32", iterations)} if dynamic else {}
    execute(after["main"].body, bindings)
    assert observed["setup"] == 1, after.script()
    assert observed["sets"], after.script()
    assert observed["sets"] == observed["waits"], after.script()
    assert all(executions == 1 for executions in observed["sets"].values()), after.script()


def test_selected_independent_fix_copies_do_not_create_distance_one_spr_ring():
    before = _scheduled_state()
    root = before["main"].body.block
    setup, loop_unit = root.body.seq
    loop = loop_unit.body
    loop = tirx.For(loop.loop_var, loop.min, loop.extent, loop.kind, loop.body,
                   annotations={"num_stages": 2})
    before = _target(kernel(seq(setup, unit(loop, core=None)),
                          buffers=root.alloc_buffers,
                          params=list(before["main"].buffer_map.values())))
    captured = []
    invocations = []

    def schedule(*args, **kwargs):
        invocations.append(1)
        deps = kwargs.get("data_deps", args[4] if len(args) > 4 else None)
        captured.extend(tuple(int(x) for x in edge) for edge in deps)
        return [0, 1], [], 2

    with TARGET, patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=schedule):
        transform.AutoSchedule()(before)
    assert invocations, "The regression must exercise loop dependency construction"
    edges = {edge[:3] for edge in captured}
    assert (0, 1, 0) not in edges and (1, 0, 1) not in edges, captured


@pytest.mark.parametrize("factory", [transform.AutoSchedule, transform.InsertSync])
@pytest.mark.parametrize("repeat", [False, True])
def test_compound_per_core_reader_cannot_duplicate_one_shot_wait(factory, repeat):
    before = _scheduled_state()
    root = before["main"].body.block
    setup = root.body.seq[0]
    selected_copies = calls(before, COPY)
    candidate = unit(tirx.Evaluate(selected_copies[0]), core=2, cost=(1, 1)).body
    if repeat:
        body = tirx.For(tirx.Var("internal", "int32"), 0, 2, tirx.ForKind.SERIAL, candidate)
    else:
        body = seq(candidate, unit(tirx.Evaluate(selected_copies[1]), core=2, cost=(1, 1)).body)
    metadata = {"core_mask": tirx.IntImm("int64", 2),
                "latency": tirx.IntImm("int64", 1), "ii": tirx.IntImm("int64", 1)}
    group = tirx.AttrStmt(metadata, "tl.ascend_per_core_task", 1, body)
    # The original outer loop still owns all copy-region index expressions.
    original_loop = root.body.seq[1].body
    loop = tirx.For(original_loop.loop_var, original_loop.min, original_loop.extent,
                   original_loop.kind, unit(group, core=None))
    before = _target(kernel(seq(setup, unit(loop, core=None)),
                          buffers=root.alloc_buffers,
                          params=list(before["main"].buffer_map.values())))
    with TARGET, pytest.raises(tvm.error.InternalError,
                               match="Fixpipe T.PerCoreTask requires one flat T.Task leaf"):
        factory()(before)


def _assert_root_fixpipe_completion(before, after, expected_consumers):
    """Prove the fixture's AIC completion paths after sync-edge reduction.

    Execute its finite control domain, then compose pipe issue order with
    actual matched set/wait completion edges. A path must contain a hard-event
    edge; source order alone cannot establish completion across pipes.
    """
    assert len(calls(before, CONFIG)) == 1
    selected = calls(before, COPY)
    selected = [call for call in selected if FLAG in call.annotations]
    assert len(selected) == 2 and all(int(call.annotations[FLAG]) == 1 for call in selected)
    analyzer = tvm.arith.Analyzer()
    executed = []

    def scalar(expr, bindings):
        value = analyzer.simplify(tirx.stmt_functor.substitute(expr, bindings))
        assert isinstance(value, tirx.IntImm), "Fixture control must evaluate to an integer"
        return int(value)

    def visit(stmt, bindings):
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child, bindings)
        elif isinstance(stmt, tirx.SBlockRealize):
            assert not stmt.iter_values
            if scalar(stmt.predicate, bindings):
                visit(stmt.block, bindings)
        elif isinstance(stmt, tirx.SBlock):
            visit(stmt.body, bindings)
        elif isinstance(stmt, tirx.AttrStmt):
            if stmt.attr_key in ("tl.ascend_task", "tl.ascend_per_core_task"):
                # This proof is about the AIC's invariant state. Mixed AIV
                # consumers retain their separate collective buffer edges.
                if "core_mask" in stmt.node and int(stmt.node["core_mask"]) == 1:
                    return
            nested = dict(bindings)
            if stmt.attr_key == "thread_extent":
                assert isinstance(stmt.node, tirx.IterVar)
                nested[stmt.node.var] = tirx.IntImm(stmt.node.var.dtype, 0)
            visit(stmt.body, nested)
        elif isinstance(stmt, tirx.For):
            start, extent = scalar(stmt.min, bindings), scalar(stmt.extent, bindings)
            step = scalar(stmt.step, bindings) if stmt.step is not None else 1
            assert step > 0 and 0 <= extent <= 8, "Keep this fixture's execution domain bounded"
            for value in range(start, start + extent, step):
                nested = dict(bindings)
                nested[stmt.loop_var] = tirx.IntImm(stmt.loop_var.dtype, value)
                visit(stmt.body, nested)
        elif isinstance(stmt, tirx.IfThenElse):
            branch = stmt.then_case if scalar(stmt.condition, bindings) else stmt.else_case
            if branch is not None:
                visit(branch, bindings)
        elif isinstance(stmt, tirx.Bind):
            bindings[stmt.var] = tirx.IntImm(stmt.var.dtype, scalar(stmt.value, bindings))
        elif isinstance(stmt, tirx.Evaluate):
            if isinstance(stmt.value, tirx.Call) and isinstance(stmt.value.op, tvm.ir.Op):
                executed.append((stmt.value, dict(bindings)))
        else:
            pytest.fail("Unexpected statement in the finite completion fixture")

    visit(after["main"].body, {})
    edges, previous, pending = defaultdict(list), {}, defaultdict(deque)
    root, consumers = [], []
    for index, (call, bindings) in enumerate(executed):
        name, pipe = call.op.name, None
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            source, destination = call.args[0].value.split("_")
            key = (source, destination, scalar(call.args[1], bindings))
            if name == "tl.ascend_set_flag":
                pipe = source
                pending[key].append(index)
            else:
                pipe = destination
                assert pending[key], "Each emitted wait needs a preceding matching set"
                edges[pending[key].popleft()].append((index, True))
        elif name in (CONFIG, "tl.ascend_set_load3d_config"):
            pipe = "S"
            if name == CONFIG:
                root.append(index)
        elif name in ("tl.tileop.ascend_load_cbuf_to_ca_3d", "tl.ascend_load_cbuf_to_ca_3d"):
            pipe = "MTE1"
        elif name in ("tl.tileop.gemm", "tl.ascend_mad"):
            pipe = "M"
        elif name == COPY:
            source = call.args[0].args[0].buffer.scope()
            destination = call.args[1].args[0].buffer.scope()
            if source == "shared.l0c" and destination == "global":
                pipe = "FIX"
                assert FLAG in call.annotations and int(call.annotations[FLAG]) == 1
                consumers.append(index)
            elif source == "shared.l1" and destination in ("shared.l0a", "shared.l0b"):
                pipe = "MTE1"
            elif source == "global" and destination == "shared.l1":
                pipe = "MTE2"
        if pipe is not None:
            if pipe in previous:
                edges[previous[pipe]].append((index, False))
            previous[pipe] = index

    assert len(root) == 1 and len(consumers) == expected_consumers
    assert not any(pending.values()), "The fixture must drain its finite hard-event protocol"
    reachable, work = set(), [(root[0], False)]
    while work:
        node, completed = work.pop()
        if (node, completed) in reachable:
            continue
        reachable.add((node, completed))
        work.extend((successor, completed or strict) for successor, strict in edges[node])
    assert all((consumer, True) in reachable for consumer in consumers), (
        "Every selected FIX consumer must be completion-reachable from its root setter"
    )
    return executed, edges


def _assert_mixed_gm_bridge_completion(mod, aic_execution):
    """Match the fixture's three typed GM channels to collective completion."""
    root = next(block for block in nodes(mod, tirx.SBlock) if block.name_hint == "tilelang_root")
    workspaces = list(root.annotations["tl.ascend.compiler_workspace_storages"])
    assert len(workspaces) == 3
    streams = {1: [], 2: []}
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        ownership = [parent for parent in parents if isinstance(parent, tirx.AttrStmt)
                     and parent.attr_key in ("tl.ascend_task", "tl.ascend_per_core_task")]
        if not ownership:
            continue
        core = int(ownership[-1].node["core_mask"])
        if core in streams:
            streams[core].append(stmt.value)

    def named(call, name):
        return isinstance(call.op, tvm.ir.Op) and call.op.name == name

    def copies(core, storage, operand):
        return [(index, call) for index, call in enumerate(streams[core])
                if named(call, COPY) and call.args[operand].args[0].buffer.data.same_as(storage)]

    channels = []
    p_reload = None
    for storage in workspaces:
        writes = [(core, index, call) for core in streams
                  for index, call in copies(core, storage, 1)]
        reads = [(core, index, call) for core in streams
                 for index, call in copies(core, storage, 0)]
        assert len(writes) == len(reads) == 1
        producer, write_index, write = writes[0]
        consumer, read_index, read = reads[0]
        assert producer != consumer
        producer_pipe = "PIPE_FIX" if producer == 2 else "PIPE_MTE3"
        sets = [call for call in streams[producer][write_index + 1:]
                if named(call, "tl.ascend_cross_core_set_flag")]
        waits = [call for call in streams[consumer][:read_index]
                 if named(call, "tl.ascend_cross_core_wait_flag")]
        assert sets and waits
        publish, acquire = sets[0], waits[-1]
        assert int(publish.args[0]) == int(acquire.args[0]) == 2
        assert publish.args[1].value == producer_pipe
        assert acquire.args[1].value == "PIPE_MTE2"
        tvm.ir.assert_structural_equal(publish.args[2], acquire.args[2])
        channels.append(int(publish.args[2]))
        if producer == 1:
            assert read.args[1].args[0].buffer.scope() == "shared.l1"
            p_reload = read
    # Flags may be safely reused after their previous channel completes.
    assert len(channels) == 3 and p_reload is not None

    # The P collective's wait precedes its GM->L1 read on MTE2. That read must
    # complete through the actual MTE2->MTE1->M event chain before the second
    # MAD can overwrite the first FIX's L0C storage.
    executed, edges = aic_execution
    reload_index = next(index for index, (call, _) in enumerate(executed) if call.same_as(p_reload))
    mads = [index for index, (call, _) in enumerate(executed) if named(call, "tl.tileop.gemm")]
    assert len(mads) == 2 and reload_index < mads[1]
    reachable, work = set(), [(reload_index, False)]
    while work:
        node, completed = work.pop()
        if (node, completed) in reachable:
            continue
        reachable.add((node, completed))
        work.extend((successor, completed or strict) for successor, strict in edges[node])
    assert (mads[1], True) in reachable, "P reload completion must precede the second MAD"


def _program(dtype):
    @T.prim_func
    def main(a: T.Tensor((16, 16), dtype), b: T.Tensor((16, 16), dtype),
             c: T.Tensor((16, 16), "float32"), d: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 16), dtype)
            b_l1 = T.alloc_l1((16, 16), dtype)
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            for i in T.serial(2):
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, c)
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, d)
    return main


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_full_pipeline_has_one_root_config_and_keeps_real_l0c_completion(dtype):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, mod, info):
            if info.name == "tl.InsertSync":
                snapshots["before"] = mod

        def run_after_pass(self, mod, info):
            if info.name == "tl.InsertSync":
                snapshots["after"] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        source = tilelang.lower(_program(dtype), target=TARGET,
                                enable_device_compile=False).kernel_source
    _assert_root_fixpipe_completion(snapshots["before"], snapshots["after"], 4)
    assert source.count("asc_set_l0c_copy_params(1, 0, 0);") == 1, source
    assert "asc_sync_notify(PIPE_M, PIPE_FIX," in source, source
    assert "asc_sync_wait(PIPE_FIX, PIPE_M," in source, source
    assert source.index("asc_set_l0c_copy_params(") < source.index("asc_copy_l0c2gm(")


def test_disabled_scheduler_keeps_per_copy_setter_and_fix_barrier():
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(_program("float16"), target=TARGET,
                                enable_device_compile=False).kernel_source
    copies = source.count("asc_copy_l0c2gm(")
    assert copies and source.count("asc_set_l0c_copy_params(1, 0, 0);") == copies, source
    assert source.count("asc_sync_pipe(PIPE_FIX);") >= copies, source
    assert "asc_sync_notify(PIPE_S, PIPE_FIX," not in source, source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_full_host_mixed_core_groups_preserve_root_config_and_gm_bridge(dtype):
    @T.prim_func
    def main(a: T.Tensor((32, 16), dtype), b: T.Tensor((16, 16), dtype),
             out: T.Tensor((32, 16), "float32")):
        with T.MixedKernel(1, sids=2):
            a_l1 = T.alloc_l1((32, 16), dtype)
            b_l1 = T.alloc_l1((16, 16), dtype)
            p_l1 = T.alloc_l1((32, 16), dtype)
            acc = T.alloc_l0c((32, 16), "float32")
            score = T.alloc_shared((32, 16), "float32")
            p = T.alloc_shared((32, 16), dtype)
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, score)
            for i, j in T.Parallel(32, 16):
                p[i, j] = T.cast(score[i, j], dtype)
            T.copy(p, p_l1)
            T.gemm(p_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, score)
            T.copy(score, out)

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, mod, info):
            if info.name == "tl.InsertSync":
                snapshots["before"] = mod

        def run_after_pass(self, mod, info):
            if info.name == "tl.InsertSync":
                snapshots["after"] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        source = tilelang.lower(main, target=TARGET, target_host="c",
                                enable_host_codegen=True,
                                enable_device_compile=False).kernel_source
    aic_execution = _assert_root_fixpipe_completion(snapshots["before"], snapshots["after"], 2)
    _assert_mixed_gm_bridge_completion(snapshots["after"], aic_execution)
    assert "__mix__(1, 2)" in source and "if ASC_IS_AIC" in source, source
    assert source.count("asc_set_l0c_copy_params(1, 0, 0);") == 1, source
    assert source.count("asc_copy_l0c2gm(") == 2, source
    assert "asc_sync_notify(PIPE_M, PIPE_FIX," in source, source
