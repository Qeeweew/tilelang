"""Release a paired GM result after each AIV's last typed GM read.

The fixture and finite DMA oracle are shared with the independent late-write
acquire tests. Both cuts must compose on the same real WAR point: the two
AIV contributions protect every old Result read, while unrelated Output
exports neither delay the release nor gain a phase-END completion certificate.
"""

from collections import defaultdict

import pytest
from tilelang import tvm
from tilelang.ascend import transform

from testing.ascend.auto_schedule.test_gm_write_acquire import (
    _acquire_stream, _completion_graph, _program,
)
from testing.ascend.auto_schedule.test_owner_input_completion import _epoch
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _name, _owner_sets, _region, _trace, _value,
)


_SET = "tl.ascend_cross_core_set_flag"
_COPY = "tl.tileop.ascend_copy"


def _release_stream(mod, lane, release_ids):
    releases, reads, tails = defaultdict(list), defaultdict(list), defaultdict(list)
    for index, (call, env, loops) in enumerate(_trace(mod, core=1, lane=lane)):
        epoch = _epoch(loops)
        if (_name(call) == _SET and epoch is not None
                and _value(call.args[2], env) in release_ids):
            releases[epoch].append((index, call.args[1].value, loops))
        elif _name(call) == _COPY:
            source, origin, _ = _region(call.args[0], env)
            destination, _, _ = _region(call.args[1], env)
            if source.name in ("Result", "ResultAlias"):
                reads[epoch].append((index, origin))
            elif destination.name == "Output":
                tails[epoch].append(index)
    return releases, reads, tails


@pytest.mark.parametrize("epochs,first", [(3, 0), (5, 0), (3, 1)])
def test_each_aiv_releases_result_at_last_read_and_keeps_paired_generations(epochs, first):
    before = _program(epochs=epochs, first=first)
    before_json = tvm.ir.save_json(before)
    after = transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == before_json

    acquisitions, prepares, exports = _acquire_stream(after)
    ticks = range(first + epochs + 2)
    assert set(acquisitions) == set(ticks)
    assert all(len(events) == 1 for events in acquisitions.values())
    release_ids = {event[1] for events in acquisitions.values() for event in events}
    assert len(release_ids) == 1, "Keep the original distance1 paired WAR clock"
    # Moving the release must not swallow the independent late-write acquire.
    for epoch, prepare in prepares.items():
        acquire, _, loops = acquisitions[epoch][0]
        positions = {var.name: value for var, value, _ in loops}
        assert prepare < acquire < min(index for index, _, _ in exports[epoch])
        assert positions["owner"] == positions["part"] == 0

    active = range(first + 1, first + epochs + 1)
    for lane in range(2):
        releases, reads, tails = _release_stream(after, lane, release_ids)
        assert set(releases) == set(ticks)
        assert all(len(events) == 1 for events in releases.values())
        assert set(reads) == set(tails) == set(active)
        for epoch in ticks:
            release, pipe, loops = releases[epoch][0]
            assert pipe == "PIPE_MTE2"
            positions = {var.name: value for var, value, _ in loops}
            if epoch in reads:
                assert [origin[1:] for _, origin in reads[epoch]] == [
                    (lane, (epoch - 1) % 2, row, 0) for row in (0, 16)]
                assert len(tails[epoch]) == 2
                assert max(index for index, _ in reads[epoch]) < release < tails[epoch][-1]
                assert positions["sub"] == 1
            else:
                assert "sub" not in positions

    trace, precedes, owner_flags = _completion_graph(after, first=first)
    reads, writes, tails, releases = (defaultdict(list) for _ in range(4))
    for index, node in enumerate(trace):
        if node.kind == "read":
            assert node.region[0] == node.lane
            assert node.region[1] == (node.epoch - 1) % 2
            reads[node.epoch].append(index)
        elif node.kind == "write":
            assert node.region[1] == node.epoch % 2
            writes[node.epoch].append(index)
        elif node.kind == "copy" and node.core == 1:
            tails[node.epoch, node.lane].append(index)
        elif (node.kind == "cross-set" and node.core == 1
              and node.flag[0] in release_ids and node.epoch is not None):
            releases[node.epoch, node.lane].append(index)

    for epoch in active:
        assert len(reads[epoch]) == 4
        for lane in range(2):
            assert len(releases[epoch, lane]) == 1
            notify = releases[epoch, lane][0]
            own_reads = [index for index in reads[epoch] if trace[index].lane == lane]
            assert len(own_reads) == len(tails[epoch, lane]) == 2
            assert all(precedes(read, notify) for read in own_reads)
            assert not precedes(tails[epoch, lane][-1], notify), \
                "Early GM release is not Vector phase-END completion"
        if epoch + 1 < first + epochs:
            assert len(writes[epoch + 1]) == 4
            # Both physical AIVs' reads must complete before either new owner
            # overwrites this same GM slot, using the actual paired protocol.
            assert all(precedes(read, write) for read in reads[epoch]
                       for write in writes[epoch + 1])
            assert all(precedes(releases[epoch, lane][0], write)
                       for lane in range(2) for write in writes[epoch + 1])
            assert all(not precedes(tails[epoch, lane][-1], write)
                       for lane in range(2) for write in writes[epoch + 1]), \
                "An unrelated Output export must not become old-Result completion"
    # Preserve owner readiness, its own-only recipients and finite ID reuse.
    for epoch in range(first, first + epochs):
        assert len(writes[epoch]) == len(reads[epoch + 1]) == 4
        for owner in range(2):
            owner_writes = [index for index in writes[epoch] if trace[index].region[0] == owner]
            owner_reads = [index for index in reads[epoch + 1] if trace[index].region[0] == owner]
            assert all(precedes(write, read) for write in owner_writes for read in owner_reads)
    assert sum(node.kind == "cross-set" and node.core == 2 and node.flag[0] in owner_flags
               for node in trace) == 2 * epochs
    for lane in range(2):
        notifications = [node for node in trace if node.kind == "cross-set" and node.core == 1
                         and node.lane == lane and node.flag[0] in release_ids]
        assert [node.epoch for node in notifications] == [None, *ticks]


@pytest.mark.parametrize("kwargs", [
    {"alias": True}, {"complete": False}, {"mismatched_activation": True},
    {"private": False}, {"distance": 0},
], ids=["distinct-buffer-view", "incomplete-owner", "different-activation",
        "no-private-proof", "unsupported-same-epoch-result"])
def test_unproven_result_keeps_wide_read_release(kwargs):
    before = _program(**kwargs)
    before_json = tvm.ir.save_json(before)
    after = transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == before_json
    assert not _owner_sets(after)
    acquisitions, _, _ = _acquire_stream(after)
    release_ids = {event[1] for events in acquisitions.values() for event in events}
    assert release_ids
    for lane in range(2):
        releases, reads, tails = _release_stream(after, lane, release_ids)
        assert reads and tails and releases
        for epoch, exports in tails.items():
            assert len(releases[epoch]) == 1
            release, _, loops = releases[epoch][0]
            assert max(exports) < release
            assert "sub" not in {var.name for var, _, _ in loops}
