"""Selected L0 reuse is an obligation checked at the public pass boundary.

Reuse the native phase fixtures and their real scheduler/completion analysis.
These CPU regressions do not compile CANN or launch a device.
"""

import re

import pytest
from tilelang import tvm
from tilelang.ascend import transform

from testing.ascend._ir import kernel
from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _pairs, _root
from testing.ascend.auto_schedule.test_l0c_phase_reuse import _admission_fixture


_MARKER = "tl.ascend_l0_phase_reuse_pairs"
_MISSING_ALIAS = "Selected L0 phase reuse lacks a completion-proven physical alias"


def _with_pairs(mod, pairs):
    root = _root(mod)
    marked = kernel(
        root.body,
        buffers=root.alloc_buffers,
        params=list(mod["main"].buffer_map.values()),
        annotations={**dict(root.annotations), _MARKER: pairs},
    )
    marked["main"] = marked["main"].with_attr(dict(mod["main"].attrs))
    return marked


def _resolve(mod):
    return transform.ResolveCore()(transform.PrepareMultiBuffer()(transform.AssignCore()(mod)))


def _mutual_alias(mod, first, second):
    aliases = _root(mod).annotations["tl.buffer_alias_map"]
    return second in aliases.get(first, []) and first in aliases.get(second, [])


@pytest.mark.parametrize("length", [0, 1, 3])
def test_selected_reuse_rejects_malformed_pair_at_insert_sync(length):
    before, cq, cp = _admission_fixture()
    scheduled = transform.AutoSchedule()(before)
    pair = [cq.data, cp.data, cq.data][:length]
    resolved = _resolve(_with_pairs(scheduled, [pair]))
    original = tvm.ir.save_json(resolved)
    with pytest.raises(tvm.error.InternalError, match="Selected L0 phase reuse requires two storage Vars per pair"):
        transform.InsertSync()(resolved)
    assert tvm.ir.save_json(resolved) == original


def test_empty_selection_keeps_conservative_capacity_at_auto_schedule():
    before, cq, cp = _admission_fixture(pressure=True)
    # With no previous choice, these two canonical C allocations can share
    # the 128 KiB bank. An explicitly empty choice must not be replanned.
    assert frozenset((cq.data, cp.data)) in _pairs(transform.AutoSchedule()(before))
    empty = _with_pairs(before, [])
    assert _MARKER in _root(empty).annotations and not _root(empty).annotations[_MARKER]
    original = tvm.ir.save_json(empty)
    with pytest.raises(tvm.error.InternalError, match=r"shared.l0c.*capacity|capacity.*shared.l0c"):
        transform.AutoSchedule()(empty)
    assert tvm.ir.save_json(empty) == original


def test_insert_sync_consumes_empty_selection_without_proving_unsafe_alias():
    before, cq, cp = _admission_fixture("outside-reader")
    scheduled = transform.AutoSchedule()(before)
    assert frozenset((cq.data, cp.data)) not in _pairs(scheduled)
    resolved = _resolve(_with_pairs(scheduled, []))
    assert _MARKER in _root(resolved).annotations and not _root(resolved).annotations[_MARKER]
    after = transform.InsertSync()(resolved)
    assert _MARKER not in _root(after).annotations
    assert not _mutual_alias(after, cq.data, cp.data)


@pytest.mark.parametrize("bad_first", [False, True], ids=["unproven-last", "unproven-first"])
def test_every_selected_pair_requires_its_own_completion_proof(bad_first):
    before, cq, cp = _admission_fixture("outside-reader")
    scheduled = transform.AutoSchedule()(before)
    good_pairs = [list(pair) for pair in _root(scheduled).annotations[_MARKER]]
    assert good_pairs and frozenset((cq.data, cp.data)) not in _pairs(scheduled)

    # Positive control: the other phases have real, bidirectional completion
    # proofs. The extra C reader escapes its phase and prevents C byte reuse.
    proven = transform.InsertSync()(_resolve(scheduled))
    assert all(_mutual_alias(proven, *pair) for pair in good_pairs)
    assert not _mutual_alias(proven, cq.data, cp.data)

    bad_pair = [cq.data, cp.data]
    pairs = [bad_pair, *good_pairs] if bad_first else [*good_pairs, bad_pair]
    resolved = _resolve(_with_pairs(scheduled, pairs))
    original = tvm.ir.save_json(resolved)
    diagnostic = re.escape(f"{_MISSING_ALIAS} between {cq.data.name} and {cp.data.name}")
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        transform.InsertSync()(resolved)
    assert tvm.ir.save_json(resolved) == original
