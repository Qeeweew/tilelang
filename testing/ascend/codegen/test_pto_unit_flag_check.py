"""PTO emits conditional native MAD controls without expanding CHECK in TIR.

These exercise source generation only. They do not establish a complete
MAD/FIX unit protocol or execute the emitted PTO program.
"""

import ast

import pytest
from tilelang import tvm
from tvm import tirx


def _source(control, params=()):
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "keys": ["pto", "ascend"]})
    # Native MAD receives flattened, typed tl.access_ptr operands after tile
    # lowering. Buffer.access_ptr's type token is not that transport format.
    a = tirx.decl_buffer((256,), "float16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((256,), "float16", name="b", scope="shared.l0b")
    c = tirx.decl_buffer((256,), "float32", name="c", scope="shared.l0c")

    def pointer(buffer, mask):
        return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [
            buffer[0], tirx.IntImm("int32", 256), tirx.IntImm("int32", mask),
        ])

    mad = tirx.Call("void", tvm.ir.Op.get("tl.ascend_mad"), [
        pointer(c, 2), pointer(a, 1), pointer(b, 1),
        16, 16, 16, control, 1, 0, True,
    ])
    body = tirx.SeqStmt([*(tirx.AllocBuffer(buffer) for buffer in (a, b, c)), tirx.Evaluate(mad)])
    func = tirx.PrimFunc(list(params), body).with_attr({
        "global_symbol": "main", "calling_conv": 2, "target": target,
    })
    build = tvm.ffi.get_global_func("target.build.tilelang_pto_without_compile")
    return build(tvm.IRModule({"main": func}), target).inspect_source()


def _executed_controls(source, **bindings):
    """Evaluate emitted branch selection, without importing or running PTO."""
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    result = []

    def condition(node):
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        assert node.func.attr == "as_logical_bool" and len(node.args) == 1
        assert isinstance(node.args[0], ast.Name)
        return bool(bindings[node.args[0].id])

    def visit(statements):
        for statement in statements:
            if isinstance(statement, ast.If):
                visit(statement.body if condition(statement.test) else statement.orelse)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if isinstance(call.func, ast.Attribute) and call.func.attr in ("mad", "mad_acc"):
                    keyword = next((item.value for item in call.keywords if item.arg == "unit_flag"), None)
                    result.append(0 if keyword is None else {"check_only": 2, "check_and_set": 3}[ast.literal_eval(keyword)])

    visit(function.body)
    return result


@pytest.mark.parametrize("control", [0, 2, 3])
@pytest.mark.parametrize("masked", [False, True], ids=["original", "check"])
def test_pto_mad_constant_unit_control(control, masked):
    expression = tirx.IntImm("int32", control)
    if masked:
        expression = tirx.bitwise_and(expression, tirx.IntImm("int32", 2))
    source = _source(expression)
    assert _executed_controls(source) == [control & 2 if masked else control]


@pytest.mark.parametrize("kind", ["select", "if_then_else"])
def test_pto_mad_internal_check_mask_preserves_conditional_branches(kind):
    enabled = tirx.Var("enabled", "bool")
    terminal = tirx.Select(enabled, 3, 0) if kind == "select" else tirx.if_then_else(enabled, 3, 0)
    expression = tirx.bitwise_and(terminal, tirx.IntImm("int32", 2))
    source = _source(expression, [enabled])
    assert _executed_controls(source, enabled=False) == [0]
    assert _executed_controls(source, enabled=True) == [2]
    assert "check_and_set" not in source


def test_pto_mad_terminal_select_with_nonterminal_check_projection():
    # This is the expression produced by LowerL1Gemm for the final full-K
    # iteration and conditional unit participation on earlier K iterations.
    enabled, last = tirx.Var("enabled", "bool"), tirx.Var("last", "bool")
    terminal = tirx.Select(enabled, 3, 0)
    control = tirx.Select(last, terminal, tirx.bitwise_and(terminal, tirx.IntImm("int32", 2)))
    source = _source(control, [enabled, last])
    for is_enabled in (False, True):
        for is_last in (False, True):
            expected = (3 if is_last else 2) if is_enabled else 0
            assert _executed_controls(source, enabled=is_enabled, last=is_last) == [expected]


@pytest.mark.parametrize("kind", ["bare", "masked", "other_mask", "dynamic_mask"])
def test_pto_mad_unknown_dynamic_control_is_still_rejected(kind):
    unknown = tirx.Var("unknown", "int32")
    enabled = tirx.Var("enabled", "bool")
    if kind == "bare":
        control, params = unknown, [unknown]
    elif kind == "masked":
        control, params = tirx.bitwise_and(unknown, 2), [unknown]
    elif kind == "other_mask":
        control, params = tirx.bitwise_and(tirx.Select(enabled, 3, 0), 3), [enabled]
    else:
        control, params = tirx.bitwise_and(tirx.Select(enabled, 3, 0), unknown), [enabled, unknown]
    with pytest.raises(tvm.error.InternalError, match="supports dynamic unit_flag_ctrl only"):
        _source(control, params)


def test_pto_check_mask_does_not_admit_unsupported_constant_leaves():
    enabled = tirx.Var("enabled", "bool")
    control = tirx.bitwise_and(tirx.Select(enabled, 1, 0), 2)
    with pytest.raises(tvm.error.InternalError, match="unsupported unit_flag_ctrl=1"):
        _source(control, [enabled])
