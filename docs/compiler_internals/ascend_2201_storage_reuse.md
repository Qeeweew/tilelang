# 2201 accumulator storage admission

AutoSchedule may repair L0C memory pressure by placing separate logical
accumulators in one physical allocation. This is a storage choice made before
scheduling, rather than a claim that asynchronous instructions have completed.
It can reduce Cube/FIX overlap in exchange for fitting the target's L0C bank.

The first implementation accepts static, rank-two FP32 buffers with the
canonical L0C layout, zero element offset, compact strides, 16-aligned matrix
dimensions and the same complete physical M pitch. Their complete logical
access intervals must be disjoint children of the same unique loop owner.
That owner and all descendant controls containing accesses must have positive
static trip counts and no conditional execution or loop breaks. Each interval
starts with a full-tile GEMM that clears the entire physical allocation view;
a nested reduction's first iteration may establish that clear. The owner
coordinate itself is never replaced by its first value when proving the clear.
Later writes must also be full-tile GEMMs. Reads use the ordinary L0C copy
contract and do not establish a definition, including empty or tail copies.

Only a capacity excess triggers this choice. Every pair in a selected group
must satisfy the compatibility proof, and the complete repair must fit the
existing target capacity. The largest allocation survives. Each logical Buffer
retains its own shape, canonical layout and typed regions; its `data` changes
to the surviving physical storage. The shared storage receives an ordinary
`tl.buffer_versions_map` entry of one, and owner eligibility references follow
the same storage identity. No buffer-version or alias side table describes a
different allocation.

Shared dependency analysis conservatively treats different Buffers with the
same `data` as conflicting. It therefore carries the previous accumulator's
FIX-read to the following GEMM-write WAR, the next epoch's reuse WAR, and WAW
hazards into both AutoSchedule and InsertSync. InsertSync establishes actual
completion before reuse. The allocator and physical footprint collector see
one allocation; MAD and FIX retain each logical view's original geometry and
the same physical M pitch.

Admission consumes the existing loop-ownership annotations. An internal
reduction whose conditional first clear has not been proved by that analysis
remains ineligible; this storage choice does not infer a replacement owner.

Explicit or already prepared version/mode or alias contracts, conflict hints, escaped pointers,
unverified views/layouts, partial definitions, conditional or dynamic owners,
explicit stages and nonzero or dynamic unit-flag hints keep their original
storage plan and resource diagnostic. Dav-3510 is unchanged. This deliberately
does not plan reuse across different memory banks or arbitrary layouts, and
does not remove later allocation-capacity validation.

The `BufferLoad` inside a direct GEMM/copy `tl.region` operand transports a
typed buffer identity; it is not a scalar read. Admission distinguishes that
identity from actual scalar loads/stores and opaque pointer uses. Region
indices, extents and call annotations remain ordinary expressions and are
checked for those unsupported uses, including an opaque use in the same task
as a precise typed operand. Scheduler read-vars includes typed storage
identities and therefore cannot serve as a scalar-escape test.
