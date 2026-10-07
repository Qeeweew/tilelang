"""Numeric unit controls do not discharge unrelated completion obligations.

These are explicit inputs to InsertSync, without scheduling, native lowering
or a device. They preserve unrelated completion obligations; they do not
prove a complete hardware unit-flag protocol.
"""

from collections import Counter

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


COPY = "tl.tileop.ascend_copy"
CONFIG = "tl.ascend_set_fixpipe_config"
ROOT_CONFIG = "tl.ascend.root_fixpipe_config"


def _target(mod, arch):
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    return tvm.IRModule({"main": mod["main"].with_attr("target", target)})


def _insert(mod):
    with mod["main"].attrs["target"]:
        return transform.InsertSync()(mod)


def _operations(mod):
    return [stmt.value for stmt in nodes(mod, tirx.Evaluate)
            if isinstance(stmt.value, tirx.Call) and isinstance(stmt.value.op, tvm.ir.Op)]


def _assert_completion_between(mod, event, producer, consumer):
    """Require an actual balanced notify/wait between this pair of tasks."""
    operations = _operations(mod)
    before = next(i for i, call in enumerate(operations) if producer(call))
    after = next(i for i, call in enumerate(operations) if consumer(call))
    assert before < after
    events = {}
    for kind, name in (("sets", "tl.ascend_set_flag"), ("waits", "tl.ascend_wait_flag")):
        events[kind] = [(i, int(tvm.arith.Analyzer().simplify(call.args[1])))
                        for i, call in enumerate(operations)
                        if call.op.name == name and call.args[0].value == event]
    assert events["sets"] and events["waits"], (event, mod.script())
    assert Counter(flag for _, flag in events["sets"]) == Counter(flag for _, flag in events["waits"])
    assert any(before < notify < wait < after and sent == consumed
               for notify, sent in events["sets"] for wait, consumed in events["waits"]), mod.script()


@pytest.mark.parametrize("unit_flag", [0, 3])
def test_unit_flag_cannot_drop_storage_free_root_fixpipe_state(unit_flag):
    # The same root S configuration / selected FIX reader contract is covered
    # by test_ascend_select_fixpipe_config. There is no buffer shared with the
    # setup task, so only its storage-free SPR RAW can establish completion.
    acc = tirx.decl_buffer((16, 16), "float32", name="acc", scope="shared.l0c")
    out = tirx.decl_buffer((16, 16), "float32", name="out")
    setup = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(CONFIG), []))
    export = copy(acc, out, annotations={ROOT_CONFIG: tirx.IntImm("int32", 1),
                                        "unit_flag_ctrl": tirx.IntImm("int32", unit_flag)})
    before = _target(kernel(seq(unit(setup, core=2), unit(export, core=2)),
                            buffers=[acc], params=[out]), "dav-2201")
    after = _insert(before)
    assert len(calls(after, CONFIG)) == len(calls(after, COPY)) == 1
    assert int(calls(after, COPY)[0].annotations["unit_flag_ctrl"]) == unit_flag
    _assert_completion_between(after, "S_FIX", lambda call: call.op.name == CONFIG,
                               lambda call: call.op.name == COPY)


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("unit_flag", [0, 3])
@pytest.mark.parametrize("reverse", [False, True], ids=["RAW", "WAR"])
def test_unit_flag_cannot_drop_non_l0c_gm_hazard(arch, unit_flag, reverse):
    acc = tirx.decl_buffer((16, 16), "float32", name="acc", scope="shared.l0c")
    shared_gm = tirx.decl_buffer((16, 16), "float16", name="bridge")
    l1 = tirx.decl_buffer((16, 16), "float16", name="reload", scope="shared.l1")
    export = copy(acc, shared_gm, annotations={"unit_flag_ctrl": tirx.IntImm("int32", unit_flag)})
    reload = copy(shared_gm, l1)
    ordered = [reload, export] if reverse else [export, reload]
    before = _target(kernel(seq(*(unit(operation, core=2) for operation in ordered)),
                            buffers=[acc, l1], params=[shared_gm]), arch)
    after = _insert(before)
    assert len(calls(after, COPY)) == 2

    def is_export(call):
        return call.op.name == COPY and call.args[0].args[0].buffer.data.same_as(acc.data)

    def is_reload(call):
        return call.op.name == COPY and call.args[0].args[0].buffer.data.same_as(shared_gm.data)

    event = "MTE2_FIX" if reverse else "FIX_MTE2"
    _assert_completion_between(after, event, is_reload if reverse else is_export,
                               is_export if reverse else is_reload)


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("unit_flag", [0, 3])
@pytest.mark.parametrize("scope", ["shared.l0c", "shared.l0c.dyn"])
def test_unqualified_l0c_controls_preserve_2201_completion(arch, unit_flag, scope):
    a = tirx.decl_buffer((16, 16), "float16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((16, 16), "float16", name="b", scope="shared.l0b")
    acc = tirx.decl_buffer((16, 16), "float32", name="acc", scope=scope)
    out = tirx.decl_buffer((16, 16), "float32", name="out")

    def mad():
        return tirx.Evaluate(T.gemm(a, b, acc, transpose_B=True, clear_accum=True,
                                   unit_flag_ctrl=unit_flag))

    export = copy(acc, out, annotations={"unit_flag_ctrl": tirx.IntImm("int32", unit_flag)})
    before = _target(kernel(seq(*(unit(operation, core=2) for operation in (mad(), export, mad()))),
                            buffers=[a, b, acc], params=[out]), arch)
    after = _insert(before)
    assert len(calls(after, "tl.tileop.gemm")) == 2 and len(calls(after, COPY)) == 1
    for event in ("M_FIX", "FIX_M"):
        sets = [call for call in calls(after, "tl.ascend_set_flag") if call.args[0].value == event]
        waits = [call for call in calls(after, "tl.ascend_wait_flag") if call.args[0].value == event]
        if unit_flag and arch == "dav-3510":
            # The 3510 compatibility path has its own hardware contract.
            assert not sets and not waits
        else:
            assert sets and waits
