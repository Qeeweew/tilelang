"""One L1 input epoch may feed several guarded, delayed readers.

The finite DMA oracle proves producer completion before every read and every
reader's completion before reusing the physical slot. These are CPU protocol
fixtures with independently specified stages, not an AutoSchedule performance
claim or an instruction timing model.
"""

from collections import defaultdict

import pytest
from tilelang import tvm
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_sync_binary_flags import (
    _completion_trace,
    _lower_protocol,
    _prove_binary_protocol,
)


def _shared_input(trips, first_prefix, versions, last_stage):
    capacity = max(1, trips)
    source = tirx.decl_buffer((capacity, 16, 16), "float16", name="Input")
    storage = tirx.decl_buffer((16, 16), "float16", name="SharedInput", scope="shared.l1")
    consumed = tirx.decl_buffer((capacity, 2, 16, 16), "float16", name="Readers", scope="shared.l0a")
    second_exists = tirx.Var("second_exists", "int32")
    epoch = tirx.Var("epoch", "int32")
    count = tirx.if_then_else(second_exists != 0, trips, first_prefix)
    producer = copy(source, storage, src_indices=[epoch, 0, 0], src_shape=[1, 16, 16])

    def reader(index):
        return copy(storage, consumed, dst_indices=[epoch, index, 0, 0], dst_shape=[1, 1, 16, 16])

    body = seq(
        unit(producer, core=2),
        unit(reader(0), core=2, stage=0, guard=epoch < first_prefix),
        unit(reader(1), core=2, stage=last_stage, guard=second_exists != 0),
    )
    loop = tirx.For(epoch, 0, count, tirx.ForKind.SERIAL, body,
                   annotations={"multi_buffer_eligible": [storage.data]})
    mod = kernel(unit(loop, core=None), buffers=[storage, consumed], params=[source, second_exists],
                 annotations={"tl.buffer_versions_map": {storage.data: versions},
                              "tl.buffer_version_mode": {storage.data: "iteration"}})
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod, second_exists


@pytest.mark.parametrize("trips", [0, 1, 2, 5])
@pytest.mark.parametrize("versions,last_stage", [(1, 0), (2, 1), (4, 3)],
                         ids=["serial-single-slot", "two-slot-delay", "four-slot-delay"])
@pytest.mark.parametrize("second_exists", [0, 1], ids=["empty-second", "two-readers"])
def test_shared_input_waits_for_every_active_reader(trips, versions, last_stage, second_exists):
    first_prefix = min(trips, trips // 2 + 1)
    before, parameter = _shared_input(trips, first_prefix, versions, last_stage)
    after = _lower_protocol(before)
    trace = _completion_trace(after, {parameter: second_exists})
    precedes = _prove_binary_protocol(trace)
    producers, consumers = {}, defaultdict(dict)
    for index, node in enumerate(trace):
        if node.kind == "produce":
            epoch, = node.epoch
            assert epoch not in producers
            producers[epoch] = index
        elif node.kind == "consume":
            epoch, reader = node.epoch
            assert reader not in consumers[epoch]
            consumers[epoch][reader] = index
    count = trips if second_exists else first_prefix
    assert set(producers) == set(range(count))
    expected = {(epoch, reader) for epoch in range(count) for reader in (0, 1)
                if (epoch < first_prefix if reader == 0 else second_exists)}
    assert {(epoch, reader) for epoch, readers in consumers.items() for reader in readers} == expected
    last_slot = {}
    for epoch, producer in producers.items():
        slot = trace[producer].slot
        for consumer in consumers[epoch].values():
            assert trace[consumer].slot == slot, "Every reader must use its producer's physical input version"
            assert precedes(producer, consumer), "Input read issued without producer completion"
        if slot in last_slot:
            previous_epoch = last_slot[slot]
            assert all(precedes(consumer, producer) for consumer in consumers[previous_epoch].values()), (
                "A shared input slot was reused before its last active reader completed"
            )
        last_slot[slot] = epoch
