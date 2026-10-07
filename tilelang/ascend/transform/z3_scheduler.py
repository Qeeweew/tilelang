"""Z3-based scheduler for auto-scheduling.

This module provides a Python implementation of the Z3 scheduler that can be
called from C++ via TVM FFI.
"""

from __future__ import annotations

import tvm_ffi
import os
import json
import time
import threading
from pathlib import Path

# Global lock to serialize Z3 calls — Z3 is not thread-safe
_z3_lock = threading.Lock()

# Try to import z3, but handle missing installation gracefully
try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("[Python Z3] WARNING: z3-solver package not installed. Z3 scheduling will not work.")


def _find_next_schedule_number(base_dir="debug"):
    """Find the smallest available number for schedule_xxx directory."""
    if not os.path.exists(base_dir):
        return 0

    existing_numbers = []
    for item in os.listdir(base_dir):
        if item.startswith("schedule_") and os.path.isdir(os.path.join(base_dir, item)):
            try:
                num = int(item[9:])  # Extract number from "schedule_xxx"
                existing_numbers.append(num)
            except ValueError:
                continue

    if not existing_numbers:
        return 0

    existing_numbers.sort()
    # Find the first gap in the sequence
    for i, num in enumerate(existing_numbers):
        if i != num:
            return i

    return len(existing_numbers)


def _convert_int_tuples(values, arity: int) -> list[tuple[int, ...]]:
    if values is None:
        return []
    result = []
    for value in values:
        if not hasattr(value, "__len__") or len(value) != arity:
            actual_arity = len(value) if hasattr(value, "__len__") else "non-sequence"
            raise ValueError(f"Expected {arity} integers per tuple, got {actual_arity}")
        result.append(tuple(int(value[i]) for i in range(arity)))
    return result


def _reduce_straight_line_orders(
    latencies, iis, resource_flags, data_deps, resource_deps,
    owner_exclusion_deps, pipe_order_deps,
):
    """Remove implied orders and choose representatives of exact graph symmetries.

    This uses only the supplied scheduling constraints. Equal resource masks
    alone do not make tasks interchangeable: all weighted data, pipe, owner,
    cost and resource neighbors must match. Swapping such tasks preserves the
    constraints and makespan, so ordering each class retains an optimal member.
    """
    n = len(latencies)
    resources = sorted({(i, j) for i, j in resource_deps if i < j})
    if any(ii < 0 for ii in iis) or any(weight < 0 for _, _, weight in data_deps):
        return [], resources, []
    incoming, outgoing = [dict() for _ in range(n)], [dict() for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    owners = [dict() for _ in range(n)]
    pipe_in, pipe_out = [set() for _ in range(n)], [set() for _ in range(n)]
    for i, j, weight in data_deps:
        outgoing[i][j] = max(outgoing[i].get(j, 0), weight)
        incoming[j][i] = outgoing[i][j]
    for i, j in resources:
        neighbors[i].add(j)
        neighbors[j].add(i)
    for i, j, forward, reverse in owner_exclusion_deps:
        # Repeated owner pairs share one Bool in the original query. Retain
        # their orientation groups as well as their weighted neighbors.
        owners[i].setdefault((j, True), set()).add((forward, reverse))
        owners[j].setdefault((i, False), set()).add((reverse, forward))
    for i, j in pipe_order_deps:
        pipe_out[i].add(j)
        pipe_in[j].add(i)

    classes = {}
    for i in range(n):
        signature = (
            latencies[i], iis[i], resource_flags[i],
            tuple(sorted(incoming[i].items())), tuple(sorted(outgoing[i].items())),
            tuple(sorted(pipe_in[i])), tuple(sorted(pipe_out[i])),
            # Closed resource neighborhoods permit mutually conflicting twins.
            tuple(sorted(neighbors[i] | {i})),
            tuple((other, tuple(sorted(weights))) for other, weights in sorted(owners[i].items())),
        )
        classes.setdefault(signature, []).append(i)
    symmetry_orders = [pair for group in classes.values() for pair in zip(group, group[1:])]

    # A data edge is a lower bound on the difference of start times; pipe and
    # symmetry orders contribute zero. On a DAG, longest paths retain the
    # exact implied lower bounds without asking Z3 to rediscover them.
    edges = [dict(row) for row in outgoing]
    for i, j in [*pipe_order_deps, *symmetry_orders]:
        edges[i][j] = max(edges[i].get(j, 0), 0)
    degrees = [0] * n
    for row in edges:
        for j in row:
            degrees[j] += 1
    ready = [i for i, degree in enumerate(degrees) if degree == 0]
    topological = []
    while ready:
        i = ready.pop()
        topological.append(i)
        for j in edges[i]:
            degrees[j] -= 1
            if degrees[j] == 0:
                ready.append(j)
    if len(topological) != n:
        # Preserve general/cyclic inputs, including genuinely inconsistent ones.
        return [], resources, []
    reach = [dict() for _ in range(n)]
    for i in reversed(topological):
        for j, weight in edges[i].items():
            reach[i][j] = max(reach[i].get(j, 0), weight)
            for k, suffix in reach[j].items():
                reach[i][k] = max(reach[i].get(k, 0), weight + suffix)

    ordered, unordered = [], []
    for i, j in resources:
        forward, reverse = reach[i].get(j), reach[j].get(i)
        if ((forward is not None and forward >= iis[i])
                or (reverse is not None and reverse >= iis[j])):
            continue
        if forward is not None and forward + iis[j] > 0:
            # The reverse disjunct is impossible, but the forward II gap may
            # still be stronger than the data/pipe path: keep that hard gap.
            ordered.append((i, j, iis[i]))
        elif reverse is not None and reverse + iis[i] > 0:
            ordered.append((j, i, iis[j]))
        else:
            unordered.append((i, j))
    return ordered, unordered, symmetry_orders


def _straight_line_makespan_bounds(latencies, iis, resource_flags, data_deps, resource_deps, owner_exclusion_deps, pipe_order_deps):
    """Prove objective bounds from the existing graph and a feasible assignment.

    These are bounds in the supplied model's units, not target measurements.
    A resource clique cannot finish sooner than its total issue work plus the
    earliest possible release and smallest remaining completion tail.
    """
    n = len(latencies)
    if any(ii < 0 for ii in iis) or any(weight < 0 for _, _, weight in data_deps):
        return 0, None
    edges = [dict() for _ in range(n)]
    for i, j, weight in [*data_deps, *((i, j, 0) for i, j in pipe_order_deps)]:
        edges[i][j] = max(edges[i].get(j, 0), weight)

    def earliest(graph):
        degrees = [0] * n
        for row in graph:
            for j in row:
                degrees[j] += 1
        ready = [i for i, degree in enumerate(degrees) if degree == 0]
        order, starts = [], [0] * n
        while ready:
            i = ready.pop()
            order.append(i)
            for j, weight in graph[i].items():
                starts[j] = max(starts[j], starts[i] + weight)
                degrees[j] -= 1
                if degrees[j] == 0:
                    ready.append(j)
        return (order, starts) if len(order) == n else ([], [])

    order, heads = earliest(edges)
    if not order:
        return 0, None
    tails = list(latencies)
    for i in reversed(order):
        for j, weight in edges[i].items():
            tails[i] = max(tails[i], weight + tails[j])
    lower = max([0, *(head + tail for head, tail in zip(heads, tails))])
    resources = {(i, j) for i, j in resource_deps if i < j}
    for bit in (1, 2, 4, 8, 16, 32, 64):
        group = [i for i in range(n) if resource_flags[i] & bit]
        # The actual supplied exclusions, including explicit API inputs, are
        # authoritative; a resource mask alone never adds a hard constraint.
        if group and all((i, j) in resources for offset, i in enumerate(group) for j in group[offset + 1:]):
            lower = max(lower, sum(iis[i] for i in group) + min(heads[i] for i in group)
                        + min(tails[i] - iis[i] for i in group))

    # Fix every disjunction in source order to construct one feasible upper
    # bound. This does not fix the optimized schedule's issue order. If that
    # orientation creates a cycle, retain only the independently proven lower.
    serial = [dict(row) for row in edges]
    for i, j in resources:
        serial[i][j] = max(serial[i].get(j, 0), iis[i])
    for i, j, forward, reverse in owner_exclusion_deps:
        if i > j:
            i, j, forward = j, i, reverse
        serial[i][j] = max(serial[i].get(j, 0), forward)
    order, starts = earliest(serial)
    upper = max([0, *(start + latency for start, latency in zip(starts, latencies))]) if order else None
    return lower, upper


def z3_schedule_python(
    latencies: list[int],
    iis: list[int],
    resource_flags: list[int],
    data_deps: list[tuple[int, int, int]],
    resource_deps: list[tuple[int, int]],
    owner_exclusion_deps: list[tuple[int, int, int, int]] | None = None,
    pipe_order_deps: list[tuple[int, int]] | None = None,
    verbose: bool = False,
) -> tuple[list[int], list[int]]:
    """Schedule a straight-line task list.

    Parameters
    ----------
    latencies : list[int]
        Task durations in the cost model's units (cycles for calibrated models,
        dimensionless ordering weights for uncalibrated models).
    iis : list[int]
        Task issue intervals in the same units as ``latencies``.
    resource_flags : list[int]
        Resource pipe mask for each task (bitmask of ResourcePipe values):
        MTE1=1, MTE2=2, MTE3=4, Cube=8, Vector=16, Fixpipe=32, Scalar=64.
    data_deps : list[tuple[int, int, int]]
        Directed ``(i, j, latency)`` constraints requiring task ``j`` to start
        at least ``latency`` cost-model units after task ``i``.
    resource_deps : list[tuple[int, int]]
        Unordered task pairs that cannot issue concurrently on one resource.
    owner_exclusion_deps : list[tuple[int, int, int, int]] | None
        Undirected shared-storage owner conflicts ``(i, j, latency_i_j,
        latency_j_i)``. Either owner may run first, but their conflicting
        storage lifetimes cannot overlap.
    pipe_order_deps : list[tuple[int, int]] | None
        Source-ordered task pairs that share a hardware pipe.
    verbose : bool
        Print solver inputs and the selected schedule.

    Returns
    -------
    tuple[list[int], list[int]]
        start_times: Start time for each task
        sorted_indices: Task indices sorted by start time
    """
    n = len(latencies)

    # For small number of tasks, return trivial schedule
    if n < 1:
        raise RuntimeError("Z3 scheduling failed: n too small")
    if pipe_order_deps is None:
        pipe_order_deps = []
    if owner_exclusion_deps is None:
        owner_exclusion_deps = []

    if verbose:
        print(f"[Python Z3] Starting scheduling for {n} tasks")
        print(f"[Python Z3] Latencies: {latencies}")
        print(f"[Python Z3] IIs: {iis}")
        print(f"[Python Z3] Resource flags: {resource_flags}")
        print(f"[Python Z3] Data dependencies: {data_deps}")
        print(f"[Python Z3] Resource dependencies: {resource_deps}")
        print(f"[Python Z3] Owner exclusions: {owner_exclusion_deps}")

    assert Z3_AVAILABLE, "z3-solver package is required but not installed"

    # Use a private Z3 context so this solve is isolated from global-context
    # state accumulated by other solves in the same process (see the loop
    # scheduler for why this is required for deterministic results).
    ctx = z3.Context()

    # Create Z3 solver
    solver = z3.Optimize(ctx=ctx)
    solver.set("rlimit", 1000000)

    # Create start time variables
    start_vars = [z3.Int(f"start_{i}", ctx) for i in range(n)]

    # Add constraints: start times must be non-negative
    for var in start_vars:
        solver.add(var >= 0)

    # Add data dependency constraints
    # For each data dependency (i, j, latency), task j must start after task i starts + latency
    for i, j, latency in data_deps:
        solver.add(start_vars[j] >= start_vars[i] + latency)
        if verbose:
            print(f"[Python Z3] Data dependency: task {j} >= task {i} + {latency}")

    ordered_resources, unordered_resources, symmetry_orders = _reduce_straight_line_orders(
        latencies, iis, resource_flags, data_deps, resource_deps,
        owner_exclusion_deps, pipe_order_deps,
    )
    for previous, current in symmetry_orders:
        solver.add(start_vars[previous] <= start_vars[current])
    for previous, current, interval in ordered_resources:
        solver.add(start_vars[current] >= start_vars[previous] + interval)

    # Add resource dependency constraints
    # For tasks i and j that use same resource, they cannot execute simultaneously
    # We create ordering variable O_i,j (True means i before j, False means j before i)
    for i, j in unordered_resources:
        if i < j:  # Only consider each pair once
            ii_i = iis[i]
            ii_j = iis[j]

            # Create ordering variable
            o_ij = z3.Bool(f"O_{i}_{j}", ctx)

            # If o_ij is True (i before j), then start_j >= start_i + ii_i
            solver.add(z3.Implies(o_ij, start_vars[j] >= start_vars[i] + ii_i))

            # If o_ij is False (j before i), then start_i >= start_j + ii_j
            solver.add(z3.Implies(z3.Not(o_ij), start_vars[i] >= start_vars[j] + ii_j))

            if verbose:
                print(f"[Python Z3] Resource dependency between {i} and {j}: ii_i={ii_i}, ii_j={ii_j}")

    for i, j, latency_i_j, latency_j_i in owner_exclusion_deps:
        i_before_j = z3.Bool(f"Owner_{i}_{j}", ctx)
        solver.add(z3.Implies(i_before_j, start_vars[j] - start_vars[i] >= latency_i_j))
        solver.add(z3.Implies(z3.Not(i_before_j), start_vars[i] - start_vars[j] >= latency_j_i))
        if verbose:
            print(f"[Python Z3] Owner exclusion between {i} and {j}: latency_i_j={latency_i_j}, latency_j_i={latency_j_i}")

    for previous, current in pipe_order_deps:
        solver.add(start_vars[previous] <= start_vars[current])

    # Objective: minimize maximum completion time (makespan)
    makespan = z3.Int("makespan", ctx)
    for i in range(n):
        latency_i = latencies[i]
        solver.add(makespan >= start_vars[i] + latency_i)
    solver.add(makespan >= 0)
    lower, upper = _straight_line_makespan_bounds(
        latencies, iis, resource_flags,
        [*data_deps, *ordered_resources], resource_deps, owner_exclusion_deps,
        [*pipe_order_deps, *symmetry_orders],
    )
    solver.add(makespan >= lower)
    if upper is not None:
        solver.add(makespan <= upper)

    # Minimize makespan
    solver.minimize(makespan)

    # Check satisfiability
    if verbose:
        print("[Python Z3] Checking satisfiability...")
    status = solver.check()
    if status == z3.sat:
        model = solver.model()

        # Extract start times
        start_times = []
        for i in range(n):
            start_time = model.eval(start_vars[i]).as_long()
            start_times.append(start_time)

        # Get makespan
        makespan_val = model.eval(makespan).as_long()

        # Sort tasks by start time (and by index as tie-breaker)
        task_indices = list(range(n))
        task_indices.sort(key=lambda idx: (start_times[idx], idx))

        if verbose:
            print(f"[Python Z3] Scheduling completed. Makespan = {makespan_val}")
        for i in range(n):
            idx = task_indices[i]
            if verbose:
                print(
                    f"[Python Z3]   Task {idx}: start_time={start_times[idx]}, "
                    f"latency={latencies[idx]}, II={iis[idx]}, "
                    f"resource_flags={resource_flags[idx]:07b}"
                )

        return start_times, task_indices
    if status == z3.unsat:
        raise RuntimeError("Z3 scheduling failed: solver returned unsat")
    raise RuntimeError(f"Z3 scheduling failed: solver returned unknown ({solver.reason_unknown()})")


# FFI-exposed function that matches C++ interface
@tvm_ffi.register_global_func("tl.transform.z3_schedule_python")
def z3_schedule_ffi(
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps,
    owner_exclusion_deps=None,
    pipe_order_deps=None,
):
    """FFI wrapper for z3_schedule_python.

    This function accepts TVM containers and converts them to Python lists.
    """
    # Convert TVM containers to Python lists
    latencies_list = list(latencies)
    iis_list = list(iis)
    resource_flags_list = list(resource_flags)

    data_deps_list = _convert_int_tuples(data_deps, 3)
    resource_deps_list = _convert_int_tuples(resource_deps, 2)
    pipe_order_deps_list = _convert_int_tuples(pipe_order_deps, 2)
    owner_exclusion_deps_list = _convert_int_tuples(owner_exclusion_deps, 4)

    # Call the actual scheduler (Z3 is not thread-safe, serialize access)
    with _z3_lock:
        start_times, _ = z3_schedule_python(
            latencies_list,
            iis_list,
            resource_flags_list,
            data_deps_list,
            resource_deps_list,
            owner_exclusion_deps=owner_exclusion_deps_list,
            pipe_order_deps=pipe_order_deps_list,
        )

    # Return only start_times, C++ side will sort by start_time
    return start_times


def _loop_issue_before(ctx, k_vars, r_vars, producer, consumer):
    """Match AutoSchedule's residue, descending-stage, source-index sort."""
    return z3.Or(
        r_vars[producer] < r_vars[consumer],
        z3.And(r_vars[producer] == r_vars[consumer], z3.Or(
            k_vars[producer] > k_vars[consumer],
            z3.And(k_vars[producer] == k_vars[consumer],
                   z3.BoolVal(producer < consumer, ctx)),
        )),
    )


def _refine_loop_prefetch(
    ctx, assertions, model, k_vars, r_vars, buffer_vars,
    prefetch_priority_pairs, buffer_sizes, *, verbose=False,
):
    """Jointly prefer a useful one-epoch load lead at the already chosen II.

    Keep every original hard assertion, including storage capacity, while
    leaving stages and versions free. A stage lead is useful only when the
    producer also issues before its consumer in the physical child list. This
    is a bounded structural preference, not a calibrated latency objective.
    """
    if not prefetch_priority_pairs:
        return model
    optimizer = z3.Optimize(ctx=ctx)
    optimizer.set("rlimit", 100000)
    optimizer.set("timeout", 100)
    optimizer.add(*assertions)
    penalties = [z3.If(z3.And(
        k_vars[consumer] >= k_vars[producer] + 1,
        _loop_issue_before(ctx, k_vars, r_vars, producer, consumer),
    ), 0, 1) for producer, consumer in prefetch_priority_pairs]
    allocated_bytes = z3.Sum([var * size for var, size in zip(buffer_vars, buffer_sizes)])
    # These auxiliary bounds cannot exclude an original feasible assignment.
    # Minimize their difference to obtain max(k)-min(k) without expanding a
    # nested max/min expression or adding a quadratic set of stage relations.
    low, high = z3.Ints("prefetch_min_k prefetch_max_k", ctx)
    optimizer.add(*[low <= stage for stage in k_vars],
                  *[high >= stage for stage in k_vars])
    objectives = [z3.Sum(penalties), allocated_bytes, high - low]
    for objective in objectives:
        optimizer.minimize(objective)
    status = optimizer.check()
    if status != z3.sat:
        if verbose:
            reason = optimizer.reason_unknown() if status == z3.unknown else str(status)
            print(f"[Python Z3 Loop] Prefetch refinement {status} ({reason}); retain feasible plan")
        return model
    candidate = optimizer.model()

    def cost(selected):
        # The original model has no auxiliary min/max variables. Evaluate the
        # same score from its actual stages rather than completing those new
        # variables with arbitrary zeros.
        stages = [selected.eval(var, model_completion=True).as_long() for var in k_vars]
        return (
            selected.eval(objectives[0], model_completion=True).as_long(),
            selected.eval(allocated_bytes, model_completion=True).as_long(),
            max(stages) - min(stages),
        )

    # A bounded optimizer may supply only a feasible incumbent. Require a
    # strict lexicographic improvement and independently check every original
    # assertion before replacing the known-feasible plan.
    if cost(candidate) >= cost(model) or any(
        not z3.is_true(candidate.eval(value, model_completion=True)) for value in assertions
    ):
        return model
    if verbose:
        print(f"[Python Z3 Loop] Prefetch preference cost: {cost(model)} -> {cost(candidate)}")
    return candidate


def _refine_loop_issue_order(
    ctx, assertions, model, k_vars, r_vars, buffer_vars, issue_priority_pairs,
    *, buffer_counts=None, prefetch_priority_pairs=None, verbose=False,
):
    """Select a better physical issue order within an already feasible plan.

    Keep the chosen stages, emitted version counts and every original assertion.
    A preference counts the actual C++ source sort: smaller residue first,
    larger stage first at equal residue, then original sibling order. It is a
    structural quality heuristic; no value here represents measured wait time.
    """
    if not issue_priority_pairs:
        return model
    optimizer = z3.Optimize(ctx=ctx)
    optimizer.set("rlimit", 100000)
    optimizer.set("timeout", 100)  # One bounded refinement per automatic loop.
    optimizer.add(*assertions)
    for var in k_vars:
        optimizer.add(var == model.eval(var, model_completion=True))
    for i, var in enumerate(buffer_vars):
        count = (buffer_counts[i] if buffer_counts is not None
                 else model.eval(var, model_completion=True))
        optimizer.add(var == count)

    # This later residue-only preference must retain useful prefetch choices
    # already selected by the joint refinement. The current model witnesses
    # each of these conditions; freezing that quality decision adds no data
    # dependency and does not replace any original completion assertion.
    for producer, consumer in prefetch_priority_pairs or []:
        useful = z3.And(
            k_vars[consumer] >= k_vars[producer] + 1,
            _loop_issue_before(ctx, k_vars, r_vars, producer, consumer),
        )
        if z3.is_true(model.eval(useful, model_completion=True)):
            optimizer.add(useful)

    penalties = []
    promoted = sorted({producer for producer, _ in issue_priority_pairs})
    for producer, consumer in issue_priority_pairs:
        before = _loop_issue_before(ctx, k_vars, r_vars, producer, consumer)
        penalties.append(z3.If(before, 0, 1))
    objectives = [z3.Sum(penalties), z3.Sum([r_vars[i] for i in promoted])]
    for objective in objectives:
        optimizer.minimize(objective)
    status = optimizer.check()
    if status != z3.sat:
        if verbose:
            reason = optimizer.reason_unknown() if status == z3.unknown else str(status)
            print(f"[Python Z3 Loop] Issue refinement {status} ({reason}); retain feasible plan")
        return model
    candidate = optimizer.model()
    def cost(selected):
        return tuple(selected.eval(value, model_completion=True).as_long()
                     for value in objectives)
    # Resource-limited optimization may yield only a feasible incumbent. Never
    # accept a modeled preference regression relative to the original plan.
    if cost(candidate) > cost(model):
        return model
    if verbose:
        print(f"[Python Z3 Loop] Issue preference cost: {cost(model)} -> {cost(candidate)}")
    return candidate


def z3_schedule_loop_python(
    num_stages: int,
    latencies: list[int],
    iis: list[int],
    resource_flags: list[int],
    data_deps: list[tuple[int, int, int, int]],  # (i, j, distance, latency)
    resource_deps: list[tuple[int, int]],
    owner_exclusion_deps: list[tuple[int, int, int, int]] | None,
    pipe_order_deps: list[tuple[int, int]] | None,
    buffer_sizes: list[int],
    memory_groups: list[list[int]],  # [[capacity, idx0, idx1, ...], ...]
    stage_order_deps: list[tuple[int, int]] | None = None,  # (u, w): k_u <= k_w
    recalculate_buffer_versions: bool = True,
    enable_offset: bool = False,
    manual_stages: list[int] | None = None,
    manual_schedule: bool = False,
    seed: int | None = 42,
    verbose: bool = False,
    stage_horizon: int | None = None,
    issue_priority_pairs: list[tuple[int, int]] | None = None,
    prefetch_priority_pairs: list[tuple[int, int]] | None = None,
) -> tuple[list[int], list[int], int]:
    """Schedule a loop body with modulo, stage, and storage constraints.

    Each start time is represented as ``k * II + r``. Directed dependencies
    constrain absolute start times, while resource and owner exclusions choose
    a non-overlapping modulo order. The solver finds the minimum feasible II by
    binary search.

    Parameters
    ----------
    num_stages : int
        Maximum number of physical versions available to automatic buffers.
    issue_priority_pairs : list[tuple[int, int]] | None
        Soft physical-source-order preferences derived from forward cross-core
        RAW handoffs. Applied only after the minimum-II solve, with its selected
        stages and emitted version counts fixed. They never replace dependencies.
    prefetch_priority_pairs : list[tuple[int, int]] | None
        Soft load-to-consumer preferences from eligible typed GM-to-on-chip
        forward RAW accesses on one concrete core and distinct pipes. Applied
        at the minimum II before fixing stages or recalculating versions; a
        preference requires both epoch lead and producer-first physical issue.
    stage_horizon : int | None
        Optional maximum number of task stages for automatic Ascend pipelines.
        Bounds the same joint solve that selects order and physical versions;
        an explicit manual schedule retains its existing stage policy.
    latencies : list[int]
        Task durations in the cost model's units (cycles for calibrated models,
        dimensionless ordering weights for uncalibrated models).
    iis : list[int]
        Task issue intervals in the same units as ``latencies``.
    resource_flags : list[int]
        Resource pipe mask for each task (bitmask of ResourcePipe values):
        MTE1=1, MTE2=2, MTE3=4, Cube=8, Vector=16, Fixpipe=32, Scalar=64.
    data_deps : list[tuple[int, int, int, int]]
        Directed ``(i, j, distance, latency)`` constraints. Negative distances
        encode automatic buffer-version variables.
    resource_deps : list[tuple[int, int]]
        Unordered task pairs that cannot issue concurrently on one resource.
    owner_exclusion_deps : list[tuple[int, int, int, int]] | None
        Undirected shared-storage owner conflicts ``(i, j, latency_i_j,
        latency_j_i)``. The chosen modulo order also constrains the reverse
        wraparound hand-off.
    pipe_order_deps : list[tuple[int, int]] | None
        Source-ordered task pairs sharing a hardware pipe.
    buffer_sizes : list[int]
        Bytes occupied by one version of each automatic buffer.
    memory_groups : list[list[int]]
        Capacity followed by member buffer indices for each memory scope.
    stage_order_deps : list[tuple[int, int]] | None
        Stage-order constraints (u, w) requiring k_u <= k_w (same or earlier
        iteration-stage).
    recalculate_buffer_versions : bool
        Recompute the minimum version counts from the selected schedule.
    enable_offset : bool
        When False, constrain each resource so that, among the tasks using that
        resource (pipe bit in resource_flags), max(start_time) - min(start_time) < II.
        This keeps all tasks on a resource within a single II window (no offsetting
        a resource's tasks across pipeline stages). When True, no such
        constraint is added. Defaults to False (constraint applied).
    manual_stages : list[int] | None
        Frontend stages for tasks in source order. Used only when
        ``manual_schedule`` is True; omitted entries are not allowed.
    manual_schedule : bool
        Preserve source issue order independently on every hardware pipe and
        constrain every task's Z3 stage to the corresponding
        ``manual_stages`` entry.
    seed : int | None
        Z3 random seed; ``None`` leaves the solver default unchanged.
    verbose : bool
        Print solver inputs, search progress, and the selected schedule.

    Returns
    -------
    tuple[list[int], list[int], int]
        start_times: Start time for each task
        buffer_versions: Number of versions for each buffer
        minimal_II: The minimal initiation interval found
    """
    n = len(latencies)

    # For small number of tasks, return trivial schedule
    if n < 1:
        raise RuntimeError("Z3 loop scheduling failed: n too small")
    if stage_horizon is not None and stage_horizon < 1:
        raise ValueError("Automatic pipeline stage_horizon must be positive")

    if issue_priority_pairs is None:
        issue_priority_pairs = []
    if prefetch_priority_pairs is None:
        prefetch_priority_pairs = []
    if stage_order_deps is None:
        stage_order_deps = []
    if pipe_order_deps is None:
        pipe_order_deps = []
    if owner_exclusion_deps is None:
        owner_exclusion_deps = []
    if manual_stages is None:
        manual_stages = [0] * n
    if manual_schedule:
        if len(manual_stages) != n:
            raise ValueError(f"Manual schedule has {len(manual_stages)} stages for {n} tasks")
        if any(stage < 0 for stage in manual_stages):
            raise ValueError(f"Manual schedule stages must be non-negative: {manual_stages}")
        if not enable_offset and any(stage != 0 for stage in manual_stages):
            raise ValueError("Non-zero manual stages require enable_offset=True")

    if verbose:
        print(f"[Python Z3 Loop] Starting scheduling for {n} tasks")
        print(f"[Python Z3 Loop] Latencies: {latencies}")
        print(f"[Python Z3 Loop] IIs: {iis}")
        print(f"[Python Z3 Loop] Resource flags: {resource_flags}")
        print(f"[Python Z3 Loop] Data dependencies with distances: {data_deps}")
        print(f"[Python Z3 Loop] Resource dependencies: {resource_deps}")
        print(f"[Python Z3 Loop] Owner exclusions: {owner_exclusion_deps}")
        print(f"[Python Z3 Loop] Stage-order dependencies: {stage_order_deps}")
        print(f"[Python Z3 Loop] Pipe-order dependencies: {pipe_order_deps}")
        print(f"[Python Z3 Loop] Manual schedule: {manual_schedule}, stages: {manual_stages}")
        print(f"[Python Z3 Loop] Buffer sizes: {buffer_sizes} within groups {memory_groups}")

    assert Z3_AVAILABLE, "z3-solver package is required but not installed"

    # Use a private Z3 context so this solve is isolated from any global-context
    # state accumulated by other solves in the same process. Sharing the global
    # context makes repeated identical solves return different (equally valid)
    # models run-to-run; a fresh context per call keeps the result deterministic.
    ctx = z3.Context()

    # Binary search for minimal II
    # Lower bound: 1 cycle
    # A sequential schedule must cover both dependency latency and the issue
    # intervals of tasks sharing a resource.
    ii_lower = max(iis)
    ii_upper = max(sum(latencies), sum(iis)) + 1
    best_ii = ii_upper
    best_model = None
    best_start_vars = None

    if verbose:
        print(f"[Python Z3 Loop] Binary search range: [{ii_lower}, {ii_upper})")

    while ii_lower < ii_upper:
        ii_mid = (ii_lower + ii_upper) // 2
        if verbose:
            print(f"[Python Z3 Loop] Testing II = {ii_mid}")

        # Create solver for feasibility check
        solver = z3.Solver(ctx=ctx)
        if seed is not None:
            solver.set("random_seed", seed)
        solver.set("rlimit", 1000000)

        k_vars = [z3.Int(f"k_{i}", ctx) for i in range(n)]
        r_vars = [z3.Int(f"r_{i}", ctx) for i in range(n)]
        start_vars = [k_vars[i] * ii_mid + r_vars[i] for i in range(n)]

        for i in range(n):
            solver.add(k_vars[i] >= 0)
            if stage_horizon is not None and not manual_schedule:
                solver.add(k_vars[i] < stage_horizon)
            solver.add(r_vars[i] >= 0)
            solver.add(r_vars[i] < ii_mid)

        if manual_schedule:
            for i, stage in enumerate(manual_stages):
                solver.add(k_vars[i] == stage)
        for previous, current in pipe_order_deps:
            solver.add(r_vars[previous] <= r_vars[current])

        buffer_vars = [z3.Int(f"buf_{i}", ctx) for i in range(len(buffer_sizes))]
        for i in range(len(buffer_sizes)):
            solver.add(buffer_vars[i] >= 1)
            solver.add(buffer_vars[i] <= num_stages)
        for group in memory_groups:
            cap = group[0]
            idxs = group[1:]
            if idxs:
                solver.add(z3.Sum([buffer_vars[i] * buffer_sizes[i] for i in idxs]) <= cap)

        # Add data dependency constraints with distance
        for u, v, distance, latency in data_deps:
            if distance >= 0:
                if verbose:
                    print(f"[Python Z3 Loop] Data dependency: task {v} - task {u} >= {latency} - {ii_mid}*{distance}")
                solver.add(start_vars[v] - start_vars[u] >= latency - ii_mid * distance)
            else:
                buffer_var_id = -distance - 1  # Convert to 0-based index
                if verbose:
                    print(
                        f"[Python Z3 Loop] Data dependency with negative distance: task {v} - task {u} >= {latency} - {ii_mid}*buf_{buffer_var_id}"
                    )
                solver.add(start_vars[v] - start_vars[u] >= latency - ii_mid * buffer_vars[buffer_var_id])

        # Add resource dependency constraints
        # For tasks i and j that use same resource, they cannot execute simultaneously
        # We create ordering variable O_i,j (True means i before j, False means j before i)
        for i, j in resource_deps:
            if i < j:  # Only consider each pair once
                ii_i = iis[i]
                ii_j = iis[j]

                # Create ordering variable
                o_ij = z3.Bool(f"O_{i}_{j}", ctx)

                solver.add(z3.Implies(o_ij, r_vars[j] - r_vars[i] >= ii_i))
                solver.add(z3.Implies(o_ij, r_vars[i] - r_vars[j] + ii_mid >= ii_j))
                solver.add(z3.Implies(z3.Not(o_ij), r_vars[i] - r_vars[j] >= ii_j))
                solver.add(z3.Implies(z3.Not(o_ij), r_vars[j] - r_vars[i] + ii_mid >= ii_i))

                if verbose:
                    print(f"[Python Z3 Loop] Resource dependency between {i} and {j}: ii_i={ii_i}, ii_j={ii_j}")

        # A physical storage ring shared by two disjoint owners is a cyclic
        # resource. Choose one modulo order and protect both the
        # forward hand-off and the wrap-around hand-off. Pipeline stage
        # offsets deliberately do not participate: owners exchange the same
        # physical slot at their issue phases, not at their logical stages.
        for i, j, latency_i_j, latency_j_i in owner_exclusion_deps:
            i_before_j = z3.Bool(f"Owner_{i}_{j}", ctx)
            solver.add(z3.Implies(i_before_j, r_vars[j] - r_vars[i] >= latency_i_j))
            solver.add(z3.Implies(i_before_j, r_vars[i] - r_vars[j] + ii_mid >= latency_j_i))
            solver.add(z3.Implies(z3.Not(i_before_j), r_vars[i] - r_vars[j] >= latency_j_i))
            solver.add(z3.Implies(z3.Not(i_before_j), r_vars[j] - r_vars[i] + ii_mid >= latency_i_j))
            if verbose:
                print(f"[Python Z3 Loop] Owner exclusion between {i} and {j}: latency_i_j={latency_i_j}, latency_j_i={latency_j_i}")

        # Add stage-order constraints for copied Let variables.
        # (u, w) requires k_u <= k_w: any unit u that reads a Let-defined var
        # must not be scheduled into a later iteration-stage than w, a unit that
        # writes the buffer the Let reads. This keeps the cloned Let (prepended
        # to the loop body in u's stage) reading the pre-write buffer value.
        for u, w in stage_order_deps:
            solver.add(k_vars[u] <= k_vars[w])
            if verbose:
                print(f"[Python Z3 Loop] Stage-order constraint: k_{u} <= k_{w}")

        # Optional offset constraint: when disabled, every task using a given
        # resource (pipe bit) must fit within a single II window, i.e.
        # max(start) - min(start) < II among that resource's tasks. For a set,
        # max - min < II is equivalent to |start_a - start_b| < II for all pairs.
        if not enable_offset:
            pipe_bits = [1, 2, 4, 8, 16, 32, 64]  # MTE1, MTE2, MTE3, Cube, Vector, Fixpipe, Scalar
            for bit in pipe_bits:
                group = [i for i in range(n) if resource_flags[i] & bit]
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        ia, ib = group[a], group[b]
                        solver.add(start_vars[ia] - start_vars[ib] < ii_mid)
                        solver.add(start_vars[ib] - start_vars[ia] < ii_mid)
                if verbose and len(group) > 1:
                    print(f"[Python Z3 Loop] Offset disabled: resource bit {bit} tasks {group} span < {ii_mid}")

        # Check feasibility
        if verbose:
            print(f"[Python Z3 Loop] Checking feasibility for II = {ii_mid}...")
        if solver.check() == z3.sat:
            if verbose:
                print(f"[Python Z3 Loop] II = {ii_mid} is feasible")
            best_ii = ii_mid
            best_model = solver.model()
            best_start_vars = start_vars
            best_k_vars = k_vars
            best_buffer_vars = buffer_vars
            best_r_vars = r_vars
            best_assertions = solver.assertions()
            # Try smaller II
            ii_upper = ii_mid
        else:
            if verbose:
                print(f"[Python Z3 Loop] II = {ii_mid} is infeasible")
            # Need larger II
            ii_lower = ii_mid + 1

    if best_model is None:
        if manual_schedule:
            raise RuntimeError(
                "Manual schedule constraints are infeasible; check per-pipe source order, T.Stage values, dependencies, and resource usage"
            )
        # No feasible II found. Fall back to a trivial sequential (non-pipelined)
        # schedule: each task starts right after the previous one finishes, in
        # index order. II = sum(latencies) makes successive loop iterations
        # non-overlapping. best_ii is clamped to >= 1 to avoid downstream
        # division by zero.
        if verbose:
            print("[Python Z3 Loop] No feasible II found; falling back to sequential schedule")
        start_times = [0] * n
        for i in range(1, n):
            start_times[i] = start_times[i - 1] + latencies[i - 1]
        best_ii = max(sum(latencies), 1)
        buffer_versions = [1] * len(buffer_sizes)
    else:
        if verbose:
            print(f"[Python Z3 Loop] Minimal feasible II = {best_ii}")

        if prefetch_priority_pairs and enable_offset and not manual_schedule:
            best_model = _refine_loop_prefetch(
                ctx, best_assertions, best_model, best_k_vars, best_r_vars,
                best_buffer_vars, prefetch_priority_pairs, buffer_sizes,
                verbose=verbose,
            )

        # Extract start times from best model. Normalize by the smallest k so the
        # earliest iteration starts at 0.
        start_times = []
        k_vars = [best_model.eval(best_k_vars[i]).as_long() for i in range(n)]
        offset = min(k_vars)
        for i in range(n):
            start_time = best_model.eval(best_start_vars[i]).as_long() - offset * best_ii
            start_times.append(start_time)

        # Extract buffer versions from best model
        if recalculate_buffer_versions:
            # if recalculation is true, we recalculate based on the schedule and dependencies
            buffer_versions = len(buffer_sizes) * [1]
            for u, v, distance, latency in data_deps:
                if distance < 0:
                    buffer_var_id = -distance - 1
                    from math import ceil

                    buffer_versions[buffer_var_id] = max(
                        buffer_versions[buffer_var_id], ceil((latency + start_times[u] - start_times[v]) / best_ii)
                    )
        else:
            buffer_versions = []
            for i in range(len(buffer_sizes)):
                buffer_var = best_model.eval(best_buffer_vars[i]).as_long()
                buffer_versions.append(buffer_var)

        if issue_priority_pairs and not manual_schedule:
            # Fix the actual emitted counts, including the existing minimum-
            # version recalculation, rather than a larger slack solver bound.
            best_model = _refine_loop_issue_order(
                ctx, best_assertions, best_model, best_k_vars, best_r_vars,
                best_buffer_vars, issue_priority_pairs,
                buffer_counts=buffer_versions,
                prefetch_priority_pairs=prefetch_priority_pairs, verbose=verbose,
            )
            # Stages and version counts stay unchanged; only residues may move.
            start_times = [best_model.eval(best_start_vars[i]).as_long() - offset * best_ii
                           for i in range(n)]

    # Sort tasks by start time (and by index as tie-breaker)
    task_indices = list(range(n))
    task_indices.sort(key=lambda idx: (start_times[idx], idx))

    if verbose:
        print(f"[Python Z3 Loop] Scheduling completed. Minimal II = {best_ii}.")
    for i in range(n):
        idx = task_indices[i]
        if verbose:
            print(
                f"[Python Z3 Loop]   Task {idx}: start_time={start_times[idx]}, "
                f"latency={latencies[idx]}, II={iis[idx]}, "
                f"resource_flags={resource_flags[idx]:07b}"
            )
    if verbose:
        for i in range(len(buffer_sizes)):
            print(f"[Python Z3 Loop]   Buffer {i}: size={buffer_sizes[i]}, num_versions={buffer_versions[i]}")

    # Save schedule visualization when verbose is True
    if verbose:
        try:
            # Find the next available schedule number
            schedule_num = _find_next_schedule_number()
            schedule_dir = Path(f"debug/schedule_{schedule_num:03d}")
            schedule_dir.mkdir(parents=True, exist_ok=True)

            # Save schedule information
            schedule_info = {
                "num_stages": num_stages,
                "num_tasks": n,
                "minimal_II": best_ii,
                "latencies": latencies,
                "iis": iis,
                "resource_flags": resource_flags,
                "data_dependencies": data_deps,
                "resource_dependencies": resource_deps,
                "owner_exclusions": owner_exclusion_deps,
                "issue_priority_pairs": issue_priority_pairs,
                "manual_schedule": manual_schedule,
                "manual_stages": manual_stages,
                "start_times": start_times,
                "sorted_indices": task_indices,
                "timestamp": time.time(),
            }
            if seed is not None:
                schedule_info["random_seed"] = seed

            # Save as JSON
            info_file = schedule_dir / "schedule_info.json"
            with open(info_file, "w") as f:
                json.dump(schedule_info, f, indent=2)

            # Directly generate matplotlib visualization (PNG and PDF)
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import colorsys
                from matplotlib.patches import Patch

                fig, ax = plt.subplots(figsize=(16, 10))
                # Set background color to light gray
                fig.patch.set_facecolor("#f0f0f0")
                ax.set_facecolor("#888888")

                # Create color map for resources (pipe mask values)
                def _get_pipe_color(mask):
                    if mask & 8:  # Cube
                        return "gold"
                    if mask & 16:  # Vector
                        return "green"
                    if mask & 2:  # MTE2 (GM->UB)
                        return "red"
                    if mask & 4:  # MTE3 (UB->GM)
                        return "orange"
                    if mask & 1:  # MTE1
                        return "cyan"
                    if mask & 32:  # Fixpipe
                        return "pink"
                    return "blue"

                # Plot each task as a horizontal bar with two phases
                for i in range(n):
                    color = _get_pipe_color(resource_flags[i])
                    start = start_times[i]
                    latency = latencies[i]
                    ii = iis[i]

                    # Calculate ii boundary
                    ii_boundary = start + ii if start + ii < start + latency else start + latency

                    # Plot [start, start+ii] phase with darker color
                    phase1_width = ii_boundary - start
                    if phase1_width > 0:
                        # Darken the color for initial phase
                        rgb = plt.cm.colors.to_rgb(color)
                        hls = colorsys.rgb_to_hls(*rgb)
                        darker_color = colorsys.hls_to_rgb(hls[0], max(0, hls[1] * 0.7), hls[2])

                        rect1 = patches.Rectangle(
                            (start, i - 0.4),  # (x, y)
                            phase1_width,
                            0.8,  # width, height
                            linewidth=1,
                            edgecolor="black",
                            facecolor=darker_color,
                            alpha=0.9,
                            hatch="//",
                            label="Initial Phase [start, start+II]" if i == 0 else "",
                        )
                        ax.add_patch(rect1)

                    # Plot [start+ii, end] phase with original color
                    phase2_width = (start + latency) - ii_boundary
                    if phase2_width > 0:
                        rect2 = patches.Rectangle(
                            (ii_boundary, i - 0.4),  # (x, y)
                            phase2_width,
                            0.8,  # width, height
                            linewidth=1,
                            edgecolor="black",
                            facecolor=color,
                            alpha=0.7,
                            label="Remaining Phase [start+II, end]" if i == 0 else "",
                        )
                        ax.add_patch(rect2)

                    # Add task label (black text for better visibility)
                    ax.text(
                        start + latency / 2,
                        i,
                        f"T{i}\nII={ii}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )

                    # Add vertical line at ii boundary
                    if phase1_width > 0 and phase2_width > 0:
                        ax.axvline(x=ii_boundary, color="white", linestyle="-", alpha=0.8, linewidth=1)

                # Set limits and labels
                max_time = max(start_times[i] + latencies[i] for i in range(n))
                min_time = min(start_times[i] for i in range(n))
                ax.set_xlim(min_time - 5, max_time + 5)
                ax.set_ylim(-1, n)
                ax.set_xlabel("Time (cost-model units)")
                ax.set_ylabel("Task Index")
                ax.set_title(f"Z3 Schedule Timeline (Minimal II={best_ii})")
                ax.grid(True, alpha=0.3)

                # Add legend for resource types and phases (outside the plot)
                legend_elements = [
                    Patch(facecolor="gold", edgecolor="black", label="Cube (M)"),
                    Patch(facecolor="green", edgecolor="black", label="Vector (V)"),
                    Patch(facecolor="red", edgecolor="black", label="MTE2 (GM->)"),
                    Patch(facecolor="orange", edgecolor="black", label="MTE3 (->GM)"),
                    Patch(facecolor="cyan", edgecolor="black", label="MTE1 (L1->L0)"),
                    Patch(facecolor="pink", edgecolor="black", label="Fixpipe"),
                    Patch(facecolor="blue", edgecolor="black", label="Other"),
                    Patch(facecolor="darkgreen", edgecolor="black", hatch="//", alpha=0.9, label="Initial Phase [start, start+II]"),
                    Patch(facecolor="green", edgecolor="black", alpha=0.7, label="Remaining Phase [start+II, end]"),
                ]
                # Place legend outside the plot on the right side
                ax.legend(
                    handles=legend_elements,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=9,
                    frameon=True,
                    framealpha=0.9,
                    facecolor="white",
                )

                # Add vertical lines for each 10 time units
                for t in range(0, max_time + 10, 10):
                    ax.axvline(x=t, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

                # Save PNG and PDF files
                png_file = schedule_dir / "schedule_timeline.png"
                pdf_file = schedule_dir / "schedule_timeline.pdf"
                plt.savefig(png_file, dpi=150, bbox_inches="tight")
                plt.savefig(pdf_file, bbox_inches="tight")
                plt.close(fig)  # Close the figure to free memory

                print(f"[Python Z3 Loop] Schedule visualization saved as PNG: {png_file}")
                print(f"[Python Z3 Loop] Schedule visualization saved as PDF: {pdf_file}")

            except ImportError as e:
                print(f"[Python Z3 Loop] Warning: matplotlib not available, skipping visualization: {e}")
            except Exception as e:
                print(f"[Python Z3 Loop] Warning: Failed to generate visualization: {e}")

            print(f"[Python Z3 Loop] Schedule visualization saved to {schedule_dir}/ (text, JSON, PNG, PDF)")

        except Exception as e:
            print(f"[Python Z3 Loop] Warning: Failed to save schedule visualization: {e}")

    return start_times, buffer_versions, best_ii


# FFI-exposed function for loop scheduling
@tvm_ffi.register_global_func("tl.transform.z3_schedule_loop_python")
def z3_schedule_loop_ffi(
    num_stages,
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps,
    owner_exclusion_deps,
    pipe_order_deps,
    buffer_sizes,
    memory_groups,
    stage_order_deps=None,
    enable_offset=False,
    manual_stages=None,
    manual_schedule=False,
    stage_horizon=0,
    issue_priority_pairs=None,
    prefetch_priority_pairs=None,
):
    """FFI wrapper for z3_schedule_loop_python.

    This function accepts TVM containers and converts them to Python lists.
    Dependency arrays are ordered as directed data dependencies, resource
    exclusions, shared-storage owner exclusions, and source pipe ordering.
    ``memory_groups`` contains ``[capacity, idx0, idx1, ...]`` entries;
    ``stage_order_deps`` contains ``(u, w)`` constraints requiring ``k_u <=
    k_w``. The remaining arguments carry the loop's offset and manual-stage
    policy from C++.
    """
    # Convert TVM containers to Python lists
    latencies_list = list(latencies)
    iis_list = list(iis)
    resource_flags_list = list(resource_flags)
    buffer_sizes_list = list(buffer_sizes)
    # Convert memory groups
    memory_groups_list = []
    if memory_groups is not None:
        for i in range(len(memory_groups)):
            memory_groups_list.append(list(memory_groups[i]))
    data_deps_list = _convert_int_tuples(data_deps, 4)
    resource_deps_list = _convert_int_tuples(resource_deps, 2)
    stage_order_deps_list = _convert_int_tuples(stage_order_deps, 2)

    manual_stages_list = []
    if manual_stages is not None:
        manual_stages_list = [int(stage) for stage in manual_stages]

    pipe_order_deps_list = _convert_int_tuples(pipe_order_deps, 2)
    owner_exclusion_deps_list = _convert_int_tuples(owner_exclusion_deps, 4)
    issue_priority_pairs_list = _convert_int_tuples(issue_priority_pairs, 2)
    prefetch_priority_pairs_list = _convert_int_tuples(prefetch_priority_pairs, 2)

    # Call the actual scheduler (Z3 is not thread-safe, serialize access)
    with _z3_lock:
        start_times, buffer_versions, best_ii = z3_schedule_loop_python(
            num_stages,
            latencies_list,
            iis_list,
            resource_flags_list,
            data_deps_list,
            resource_deps_list,
            owner_exclusion_deps_list,
            pipe_order_deps_list,
            buffer_sizes_list,
            memory_groups_list,
            stage_order_deps=stage_order_deps_list,
            enable_offset=bool(enable_offset),
            manual_stages=manual_stages_list,
            manual_schedule=bool(manual_schedule),
            seed=42,
            stage_horizon=int(stage_horizon) if stage_horizon else None,
            issue_priority_pairs=issue_priority_pairs_list,
            prefetch_priority_pairs=prefetch_priority_pairs_list,
        )

    # Return start_times and buffer_versions as separate arrays for easier FFI handling
    # C++ side expects a tuple of (start_times_array, buffer_versions_array, best_ii)
    return (start_times, buffer_versions, best_ii)
