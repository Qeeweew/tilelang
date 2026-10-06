/*!
 * \file tl/ascend/transform/attr.h
 * \brief Attributes owned by Ascend lowering and scheduling.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_ATTR_H_
#define TVM_TL_ASCEND_TRANSFORM_ATTR_H_

namespace tvm {
namespace tl {
namespace attr {

// A user-authored logical Ascend per-core task whose dependency-bearing
// statements all issue on one hardware pipe. AutoSchedule schedules it once,
// then expands synchronization back to its guarded candidate sites.
constexpr const char *kAscendPerCoreTask = "tl.ascend_per_core_task";

// Groups statements into one AutoSchedule TaskNode. Inside a
// kAscendPerCoreTask region, each marker is one concrete candidate. This marker
// carries task semantics and compiler-internal core ownership; stage metadata
// lives on kScheduleUnit.
constexpr const char *kAscendTask = "tl.ascend_task";

// Assigns one frontend-requested software-pipeline stage to every scheduler
// task materialized from the enclosed statements. Consumed by
// MaterializeScheduleUnits before AutoSchedule.
constexpr const char *kAscendStage = "tl.ascend_stage";

// Short-lived scheduled-TIR wrapper shared by AutoSchedule, AssignCore,
// PrepareMultiBuffer, ResolveCore, InsertSync, MaterializeMultiBuffer, and
// LowerScheduledTIR. It carries stage/guard metadata without introducing task
// grouping or core-placement semantics.
constexpr const char *kScheduleUnit = "tl.schedule_unit";

// Root Array<Var> provenance for compiler-created global handoff allocations.
// Regions carry ownership and dependencies; this set only distinguishes
// versionable workspace from external tensors and user-created global storage.
constexpr const char *kCompilerWorkspaceStorages =
    "tl.ascend.compiler_workspace_storages";

// A logical zero fill created by GM handoff lowering. Storage normalization
// extends complete matrix axes to their physical padding and consumes this
// intent before scheduling; partial slices and empty domains stay bounded.
constexpr const char *kInitializeL1Padding = "tl.ascend.initialize_l1_padding";

} // namespace attr
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_ATTR_H_
