"""Automatic prefetch selects useful epoch lead under the original constraints.

Solver costs below are dimensionless diagnostic weights, not device timings.
The integration checks stop at CPU lowering and never launch a device kernel.
"""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

z3 = pytest.importorskip("z3")
from tilelang.ascend.transform import z3_scheduler


def _values(model, variables):
    return [model.eval(variable, model_completion=True).as_long() for variable in variables]


def _physical_order(model, k, r):
    stages, residues = _values(model, k), _values(model, r)
    return sorted(range(len(k)), key=lambda index: (residues[index], -stages[index], index))


def _useful_pairs(model, k, r, pairs):
    stages = _values(model, k)
    order = _physical_order(model, k, r)
    return [(producer, consumer) for producer, consumer in pairs
            if stages[consumer] >= stages[producer] + 1
            and order.index(producer) < order.index(consumer)]


def _plan(*, capacity=512 * 1024, num_stages=3, horizon=3):
    # Two GM loads feed one native sub-K compound. The reverse edges protect
    # the last reads before the next GM write reuses each L1 physical slot.
    ctx, ii = z3.Context(), 11
    k = [z3.Int(f"k_{index}", ctx) for index in range(3)]
    r = [z3.Int(f"r_{index}", ctx) for index in range(3)]
    buffers = [z3.Int(f"buf_{index}", ctx) for index in range(2)]
    sizes = [64 * 1024, 64 * 1024]
    start = [stage * ii + phase for stage, phase in zip(k, r)]
    hard = z3.Solver(ctx=ctx)
    for stage, phase in zip(k, r):
        hard.add(stage >= 0, stage < horizon, phase >= 0, phase < ii)
    for variable in buffers:
        hard.add(variable >= 1, variable <= num_stages)
    hard.add(z3.Sum([variable * size for variable, size in zip(buffers, sizes)]) <= capacity)
    hard.add(start[2] - start[0] >= 1, start[2] - start[1] >= 0)
    hard.add(start[0] - start[2] >= 5 - ii * buffers[0])
    hard.add(start[1] - start[2] >= 7 - ii * buffers[1])
    # Preserve both modulo resource exclusion and an independent stage-order
    # fact. The refinement is not allowed to replace either original contract.
    hard.add(z3.Or(z3.And(r[1] - r[0] >= 1, r[0] - r[1] + ii >= 1),
                   z3.And(r[0] - r[1] >= 1, r[1] - r[0] + ii >= 1)))
    hard.add(k[0] <= k[1])
    chosen = z3.Solver(ctx=ctx)
    chosen.add(*hard.assertions(), *[stage == 0 for stage in k])
    chosen.add(r[0] == 0, r[1] == 1, r[2] == 1, *[variable == 1 for variable in buffers])
    assert chosen.check() == z3.sat
    return ctx, hard, chosen.model(), k, r, buffers, sizes


def _refine(plan, *, pairs=((0, 2), (1, 2))):
    ctx, hard, before, k, r, buffers, sizes = plan
    return z3_scheduler._refine_loop_prefetch(
        ctx, hard.assertions(), before, k, r, buffers, pairs, sizes,
    )


def test_prefetch_selects_stage_lead_and_physical_producer_first_together():
    plan = _plan()
    _, hard, before, k, r, buffers, _ = plan
    assert _useful_pairs(before, k, r, [(0, 2), (1, 2)]) == []
    after = _refine(plan)
    assert _useful_pairs(after, k, r, [(0, 2), (1, 2)]) == [(0, 2), (1, 2)]
    assert _values(after, buffers) == [2, 2]
    stages = _values(after, k)
    assert max(stages) - min(stages) == 1  # Preference saturates after one epoch.
    assert all(z3.is_true(after.eval(assertion, model_completion=True))
               for assertion in hard.assertions())


@pytest.mark.parametrize("restriction", ["capacity", "versions", "horizon"])
def test_unavailable_prefetch_keeps_one_slot_and_original_constraints(restriction):
    plan = _plan(capacity=128 * 1024 if restriction == "capacity" else 512 * 1024,
                 num_stages=1 if restriction == "versions" else 3,
                 horizon=1 if restriction == "horizon" else 3)
    _, hard, _, k, r, buffers, _ = plan
    after = _refine(plan)
    assert _useful_pairs(after, k, r, [(0, 2), (1, 2)]) == []
    assert _values(after, buffers) == [1, 1]
    assert all(z3.is_true(after.eval(assertion, model_completion=True))
               for assertion in hard.assertions())


def test_equal_residue_with_later_consumer_stage_is_not_producer_first():
    # C++ sorts equal residues by decreasing stage. Merely selecting k_c>k_p
    # at the same residue issues the consumer first and cannot earn prefetch.
    ctx = z3.Context()
    k = [z3.Int("k0", ctx), z3.Int("k1", ctx)]
    r = [z3.Int("r0", ctx), z3.Int("r1", ctx)]
    buffer = z3.Int("buf", ctx)
    hard = z3.Solver(ctx=ctx)
    hard.add(k[0] == 0, k[1] == 1, r[0] == 0, r[1] >= 0, r[1] < 4, buffer == 1)
    chosen = z3.Solver(ctx=ctx)
    chosen.add(*hard.assertions(), r[1] == 0)
    assert chosen.check() == z3.sat
    before = chosen.model()
    assert _physical_order(before, k, r) == [1, 0]
    after = z3_scheduler._refine_loop_prefetch(
        ctx, hard.assertions(), before, k, r, [buffer], [(0, 1)], [1],
    )
    assert _useful_pairs(after, k, r, [(0, 1)]) == [(0, 1)]
    assert _values(after, r)[1] > 0


@pytest.mark.parametrize("status", [z3.unknown, z3.unsat])
def test_refinement_exhaustion_or_unsat_retains_original_model(monkeypatch, status):
    original = z3.Optimize

    class NoCandidate:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def check(self):
            return status

        def reason_unknown(self):
            return "test resource exhaustion"

    plan = _plan()
    monkeypatch.setattr(z3, "Optimize", NoCandidate)
    assert _refine(plan) is plan[2]


@pytest.mark.parametrize("candidate", ["worse", "equal", "violates-hard-constraint"])
def test_untrusted_incumbent_must_be_strictly_better_and_legal(monkeypatch, candidate):
    plan = _plan()
    good = _refine(plan)
    ctx, hard, original_model, k, r, buffers, sizes = plan
    assert _useful_pairs(good, k, r, [(0, 2), (1, 2)]) == [(0, 2), (1, 2)]
    selected = good if candidate == "equal" else original_model
    if candidate == "violates-hard-constraint":
        # A tempting zero-byte model must be rejected even if its structural
        # preferences match. Resource limits are never a license to trust it.
        illegal = z3.Solver(ctx=ctx)
        illegal.add(*[variable == good.eval(variable) for variable in [*k, *r]])
        illegal.add(*[variable == 0 for variable in buffers])
        assert illegal.check() == z3.sat
        selected = illegal.model()
    original = z3.Optimize

    class Incumbent:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def check(self):
            return z3.sat

        def model(self):
            return selected

    monkeypatch.setattr(z3, "Optimize", Incumbent)
    after = z3_scheduler._refine_loop_prefetch(
        ctx, hard.assertions(), good, k, r, buffers, [(0, 2), (1, 2)], sizes,
    )
    assert after is good


def test_no_prefetch_pairs_preserves_original_model():
    plan = _plan()
    assert _refine(plan, pairs=[]) is plan[2]


def test_later_handoff_issue_refinement_preserves_selected_prefetch():
    plan = _plan()
    before = _refine(plan)
    ctx, hard, _, k, r, buffers, _ = plan
    pairs = [(0, 2), (1, 2)]
    # A conflicting handoff preference must not undo the already selected
    # producer-first prefetch while changing only the fixed plan's residues.
    after = z3_scheduler._refine_loop_issue_order(
        ctx, hard.assertions(), before, k, r, buffers, [(2, 0)],
        buffer_counts=[2, 2], prefetch_priority_pairs=pairs,
    )
    assert _useful_pairs(after, k, r, pairs) == pairs
    assert all(z3.is_true(after.eval(assertion, model_completion=True))
               for assertion in hard.assertions())


def _loop_inputs(**changes):
    values = dict(
        num_stages=3, latencies=[1, 1, 11], iis=[1, 1, 11], resource_flags=[2, 2, 9],
        data_deps=[(0, 0, -1, 1), (0, 2, 0, 1), (1, 1, -2, 1),
                   (1, 2, 0, 0), (2, 0, -1, 5), (2, 1, -2, 7)],
        resource_deps=[(0, 1)], owner_exclusion_deps=[], pipe_order_deps=[],
        buffer_sizes=[64 * 1024, 64 * 1024], memory_groups=[[512 * 1024, 0, 1]],
        enable_offset=True, stage_horizon=3,
    )
    values.update(changes)
    return values


def test_prefetch_refinement_preserves_minimum_ii_and_emits_required_versions():
    baseline = z3_scheduler.z3_schedule_loop_python(**_loop_inputs())
    starts, versions, ii = z3_scheduler.z3_schedule_loop_python(
        **_loop_inputs(prefetch_priority_pairs=[(0, 2), (1, 2)]),
    )
    assert baseline[2] == ii == 11  # Equals the unchanged task-II lower bound.
    assert versions == [2, 2]
    stages = [start // ii for start in starts]
    physical = sorted(range(3), key=lambda index: (starts[index] % ii, -stages[index], index))
    assert physical.index(0) < physical.index(2) and physical.index(1) < physical.index(2)
    assert stages[2] >= max(stages[:2]) + 1


@pytest.mark.parametrize("mode", ["manual", "no-offset"])
def test_explicit_schedule_controls_bypass_prefetch_refinement(monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("Prefetch refinement must preserve explicit scheduling policy")

    monkeypatch.setattr(z3_scheduler, "_refine_loop_prefetch", forbidden)
    options = dict(manual_schedule=True, manual_stages=[0, 0, 0]) if mode == "manual" else dict(enable_offset=False)
    _, versions, ii = z3_scheduler.z3_schedule_loop_python(
        **_loop_inputs(prefetch_priority_pairs=[(0, 2), (1, 2)], **options),
    )
    assert ii == 11 and versions == [1, 1]


@pytest.mark.parametrize("tile_n", [128, 256])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_ordinary_product_gemm_prefetches_both_l1_operands(tile_n, dtype):
    import tilelang
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import calls, nodes

    path = Path(__file__).resolve().parents[3] / "examples/ascend/gemm_910b.py"
    spec = importlib.util.spec_from_file_location("_prefetch_product_gemm", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    program = example.gemm(128, tile_n, 1024, dtype=dtype, tile_n=tile_n)
    snapshots = []

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AutoSchedule":
                snapshots.append(mod)

    with tvm.transform.PassContext(instruments=[Capture()]):
        tilelang.lower(program, target={"kind": "ascend", "arch": "dav-2201"},
                       enable_host_codegen=False, enable_device_compile=False)
    assert snapshots
    after, = snapshots[-1].functions.values()
    outer, = [loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "ki"]
    children = list(outer.body.seq) if isinstance(outer.body, tirx.SeqStmt) else [outer.body]
    roles, stages = [], {}
    for child in children:
        assert isinstance(child, tirx.AttrStmt) and child.attr_key == "tl.schedule_unit", after.script()
        nested = nodes(child.body, tirx.For)
        if nested:
            assert len(nested) == 1 and nested[0].loop_var.name.startswith("gemm_sub_k_"), after.script()
            role = "child"
        else:
            movement, = calls(child.body, "tl.tileop.ascend_copy")
            source, destination = movement.args
            assert source.args[0].buffer.scope() == "global", after.script()
            buffer = destination.args[0].buffer
            assert buffer.scope() == "shared.l1" and buffer.name in ("a_l1", "b_l1"), after.script()
            role = buffer.name
        roles.append(role)
        stages[role] = int(child.node["stage"])
    assert set(roles) == {"a_l1", "b_l1", "child"}, after.script()
    assert roles.index("a_l1") < roles.index("child") and roles.index("b_l1") < roles.index("child"), after.script()
    assert stages["child"] >= max(stages["a_l1"], stages["b_l1"]) + 1, after.script()
    root, = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root"]
    versions = root.annotations["tl.buffer_versions_map"]
    for scope in ("shared.l1", "shared.l0a", "shared.l0b"):
        buffers = [buffer for buffer in root.alloc_buffers if buffer.scope() == scope]
        assert buffers, after.script()
        assert all(int(versions[buffer.data]) == 2 for buffer in buffers), after.script()


def _boundary_fixture(kind):
    from tilelang import tvm
    from tilelang.ascend import transform
    from tvm import tirx
    from testing.ascend._ir import calls, copy, kernel, nodes, seq
    from testing.ascend.auto_schedule._scheduled_ir import unit

    source = tirx.decl_buffer((4, 32), "float32", name="source")
    l1 = tirx.decl_buffer((32,), "float32", name="l1", scope="shared.l1")
    extra_l1 = tirx.decl_buffer((32,), "float32", name="extra_l1", scope="shared.l1")
    l0 = tirx.decl_buffer((32,), "float32", name="l0", scope="shared.l0a")
    ub = tirx.decl_buffer((32,), "float32", name="ub", scope="shared.dyn")
    i, inner = tirx.Var("epoch", "int32"), tirx.Var("inner", "int32")
    producer = unit(copy(source, l1, src_indices=[i, 0], src_shape=[1, 32]), core=2, stage=-1, cost=(1, 1))
    consumer = unit(copy(l1, l0), core=2, stage=-1, cost=(3, 1))
    if kind == "rmw":
        consumer = unit(tirx.BufferStore(l1, l1[0] + 1, [0]), core=2, stage=-1, cost=(3, 1))
    elif kind in ("mixed-child", "overlap-pipe"):
        # The child really reads the producer's L1 storage, but either owns
        # both C/V cores or also contains MTE2 work. Neither is a candidate.
        extra = unit(copy(source, ub if kind == "mixed-child" else extra_l1,
                          src_indices=[i, 0], src_shape=[1, 32]),
                     core=1 if kind == "mixed-child" else 2, stage=-1, cost=(1, 1))
        consumer = unit(tirx.For(inner, 0, 2, tirx.ForKind.SERIAL,
                                 seq(consumer, extra)), core=None, stage=-1)
    buffers, params, eligible = [l1, extra_l1, l0, ub], [source], [l1.data]
    if kind == "cross-core":
        # The ordinary Cube->GM->Vector handoff is a completion dependency,
        # not a same-core GM-to-local producer eligible for this preference.
        accumulator = tirx.decl_buffer((32,), "float32", name="accumulator", scope="shared.l0c")
        workspace = tirx.decl_buffer((32,), "float32", name="workspace")
        producer = unit(copy(accumulator, workspace), core=2, stage=-1, cost=(1, 1))
        consumer = unit(copy(workspace, ub), core=1, stage=-1, cost=(3, 1))
        buffers.append(accumulator)
        params.append(workspace)
        eligible = [ub.data]
    elif kind in ("constant-header", "storage-read-header"):
        output = tirx.decl_buffer((4, 32), "float32", name="output")
        minimum = (tirx.Cast("int32", ub[0])
                   if kind == "storage-read-header" else tirx.IntImm("int32", 0))
        leaf = unit(copy(source, ub, src_indices=[i, 0], src_shape=[1, 32]),
                    core=1, stage=-1, cost=(1, 1))
        producer = unit(tirx.For(inner, minimum, 2, tirx.ForKind.SERIAL, leaf),
                        core=None, stage=-1)
        consumer = unit(copy(ub, output, dst_indices=[i, 0], dst_shape=[1, 32]),
                        core=1, stage=-1, cost=(3, 1))
        params.append(output)
        eligible = [ub.data]

        # The paired inputs differ only in this real control-header read.
        # Its typed-copy leaf remains GM->UB, with no read of UB itself.
        movement, = calls(producer, "tl.tileop.ascend_copy")
        assert movement.args[0].args[0].buffer.data.same_as(source.data)
        assert movement.args[1].args[0].buffer.data.same_as(ub.data)
        control, = nodes(producer, tirx.For)
        header_reads = nodes(control.min, tirx.BufferLoad)
        assert bool(header_reads) == (kind == "storage-read-header")
        assert all(read.buffer.data.same_as(ub.data) for read in header_reads)
        assert int(control.extent) == 2
    annotations = {"multi_buffer_eligible": eligible, "tl.ascend_auto_pipeline": True, "num_stages": 3}
    loop = unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, seq(producer, consumer), annotations=annotations), core=None, stage=-1)
    mod = kernel(loop, buffers=buffers, params=params)
    arch = "dav-3510" if kind == "3510" else "dav-2201"
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return transform.AutoSchedule()(mod)


def test_prefetch_candidates_require_2201_concrete_core_disjoint_pipe_and_plain_raw():
    # C++ caches the callback after first use. Isolate the observer from every
    # other test and exercise actual typed-copy/dependency candidate discovery.
    script = r'''
import tvm_ffi
from testing.ascend.auto_schedule.test_prefetch_selection import _boundary_fixture
original = tvm_ffi.get_global_func("tl.transform.z3_schedule_loop_python")
seen = []
problems = []
def observe(*args):
    assert len(args) == 18, len(args)
    assert list(map(int, args[17])) == [1] * len(args[8])
    seen.append([tuple(map(int, pair)) for pair in args[16]])
    problems.append({
        "pipes": tuple(map(int, args[3])),
        "data": tuple(tuple(map(int, edge))[:3] for edge in args[4]),
        "owners": tuple(tuple(map(int, edge)) for edge in args[6]),
        "buffer_sizes": tuple(map(int, args[8])),
    })
    return original(*args)
tvm_ffi.register_global_func("tl.transform.z3_schedule_loop_python", observe, override=True)
header_problems = {}
for kind in ("eligible", "3510", "rmw", "mixed-child", "overlap-pipe", "cross-core",
             "constant-header", "storage-read-header"):
    seen.clear()
    problems.clear()
    _boundary_fixture(kind)
    assert seen, kind
    # Nested child callbacks may occur first; the outer loop is always last.
    assert seen[-1] == ([(0, 1)] if kind in ("eligible", "constant-header") else []), (kind, seen)
    if kind in ("constant-header", "storage-read-header"):
        outer = problems[-1]
        # AIV:MTE2 and AIV:MTE3, one outer-owned UB slot candidate, and the
        # same forward data edge: neither core/pipe nor owner rejection is
        # allowed to substitute for the whole-node storage-role check.
        assert outer["pipes"] == (2 << 7, 4 << 7), (kind, outer)
        assert outer["buffer_sizes"] == (32 * 4,), (kind, outer)
        assert outer["owners"] == (), (kind, outer)
        assert (0, 1, 0) in outer["data"], (kind, outer)
        header_problems[kind] = outer
assert header_problems["constant-header"] == header_problems["storage-read-header"], header_problems
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
