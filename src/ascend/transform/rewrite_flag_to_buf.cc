/*!
 * \file rewrite_flag_to_buf.cc
 * \brief Normalize each hard_event's synchronization into the 8-slot flag
 * namespace, spilling excess blocks to the shared asc_lock/asc_unlock mutex
 * pool.
 *
 * A set_flag/wait_flag event-pair (hard_event) owns only 8 event_id slots per
 * physical core. On dav-2201, concrete CUBE/VECTOR regions are normalized
 * independently. If any call for an event has unknown/shared ownership, that
 * entire event retains the conservative function-wide union. A
 * sparse layout may use at most 8 slots while still containing an out-of-range
 * id, so every such layout is compacted into [0,8). When a hard_event's sync
 * points need more than 8 slots in total, dav-3510 can spill to the shared
 * 32-slot asc_lock/asc_unlock mutex pool; dav-2201 must reject the overflow.
 * To waste as few flag slots as possible, we
 * treat each sync point as an indivisible block (a contiguous event_id range of
 * size = its version count, since a dynamic event_id = iter%nv + base cannot be
 * split across the 8-boundary at runtime) and run a 0/1 knapsack (capacity 8)
 * per hard_event: the subset of blocks that fills the 8 flag slots best is KEPT
 * as set_flag/wait_flag (renumbered into [0,8)); the rest SPILL to the mutex
 * pool.
 *
 * Spill lowering (hard_event = "PROD_CONS", split at '_'):
 *   set_flag<PROD_CONS>(id):  asc_lock(PIPE_PROD, buf, ASC_LOCK_NON_BLOCK);
 *                             asc_unlock(PIPE_PROD, buf, ASC_LOCK_NON_BLOCK);
 *   wait_flag<PROD_CONS>(id): asc_lock(PIPE_CONS, buf, ASC_LOCK_BLOCK);
 *                             asc_unlock(PIPE_CONS, buf, ASC_LOCK_BLOCK);
 *
 * A hard_event whose sync points are already within [0,8) is unchanged. A
 * sparse hard_event with at most 8 live slots but an out-of-range id is
 * compacted without spilling.
 *
 * Must run after InferBufferAliases for manual schedules, because that pass
 * relies on set_flag/wait_flag as liveness-graph anchors and cannot model
 * asc_lock/asc_unlock. The standard pipeline places this after
 * MergeUBAllocations.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ascend/op/builtin.h"
#include "ascend/target_utils.h"
#include "core_mask.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// Total shared asc_lock/asc_unlock mutex slots available on the hardware.
constexpr int kMaxBufId = 32;

// Number of event_id slots a single set_flag/wait_flag event-pair supports.
constexpr int kFlagSlotsPerPair = 8;

// gemm_l1 / blockscaled_gemm_l1 templates hold a 2-slot double-buffer window
// [buf_offset, buf_offset+1] in the mutex pool (see src/tl_templates/ascend/
// gemm.h, sk & 1). Spilled flags must not reuse those slots.
constexpr int kGemmL1BufSlots = 2;

// Arg index of the buf_offset operand in each gemm_l1 intrinsic call.
constexpr int kGemmL1BufOffsetArg = 10;
constexpr int kBlockscaledGemmL1BufOffsetArg = 14;

// Split "PROD_CONS" -> {"PIPE_PROD", "PIPE_CONS"}.
std::pair<std::string, std::string> SplitHardEvent(const std::string &he) {
  size_t pos = he.find('_');
  ICHECK(pos != std::string::npos)
      << "RewriteFlagToBuf: malformed hard_event '" << he << "'";
  return {"PIPE_" + he.substr(0, pos), "PIPE_" + he.substr(pos + 1)};
}

std::optional<std::string> GetFlagHardEvent(const CallNode *call) {
  if (call == nullptr)
    return std::nullopt;
  if (!call->op.same_as(ascend_set_flag()) &&
      !call->op.same_as(ascend_wait_flag()))
    return std::nullopt;
  ICHECK_EQ(call->args.size(), 2u)
      << "RewriteFlagToBuf: set_flag/wait_flag expects 2 args";
  const auto *he = call->args[0].as<StringImmNode>();
  ICHECK(he) << "RewriteFlagToBuf: hard_event arg must be a StringImm";
  return std::string(he->value);
}

// A sync-point block: a contiguous [lo, hi] range of event_ids that must be
// placed together (all in flags, or all in the mutex pool).
struct Block {
  int64_t lo;
  int64_t hi;
  int64_t size() const { return hi - lo + 1; }
};

// Resolution for one block: whether it stays as flags, and the delta to add to
// the original event_id (new_id = event_id + delta).
struct BlockAssign {
  int64_t lo;
  int64_t hi;
  bool kept;
  int64_t delta;
};

using EventIdBounds = std::pair<int64_t, int64_t>;
// An unassigned core denotes the original conservative function-wide union,
// never an additional physical bank.
using FlagNamespace = std::pair<std::string, CoreMask>;
struct FlagCallRange {
  EventIdBounds bounds;
  FlagNamespace name_space;
};
using FlagCallRanges =
    std::unordered_map<const CallNode *, std::vector<FlagCallRange>>;
using FlagAssignments = std::map<FlagNamespace, std::vector<BlockAssign>>;

// Populate the complete lexical context so const_int_bound resolves dynamic
// event_ids through enclosing loops, flat Bind declarations, branch guards,
// and assumptions.
EventIdBounds EventIdRange(const ConstrSet &constraints, const std::string &he,
                           const PrimExpr &event_id) {
  arith::Analyzer analyzer;
  constraints.Populate(analyzer);
  arith::ConstIntBound bound = analyzer.const_int_bound(event_id);
  ICHECK(bound->min_value != arith::ConstIntBound::kNegInf &&
         bound->max_value != arith::ConstIntBound::kPosInf)
      << "RewriteFlagToBuf: cannot bound event_id for hard_event '" << he
      << "'";
  ICHECK_GE(bound->min_value, 0)
      << "RewriteFlagToBuf: negative event_id for hard_event '" << he << "'";
  return {bound->min_value, bound->max_value};
}

// Phase A: collect, per hard_event, the raw event_id ranges of every flag call.
// Also tracks the highest buf_id reserved by gemm_l1 intrinsics so spilled
// flags can start allocating past them.
class FlagUsageCollector : public ConstrVisitor {
public:
  std::map<FlagNamespace, std::vector<EventIdBounds>> ranges;
  std::vector<FlagNamespace> order; // first-encounter order
  // Preserve the exact range proven at each call site so Phase B does not
  // repeat the analysis in a different traversal context.
  FlagCallRanges call_ranges;
  // Highest buf_id reserved by gemm_l1 + 1 (0 if no gemm_l1 present).
  int64_t reserved_hwm = 0;

  void FinalizeNamespaces(bool split_core_banks) {
    std::unordered_set<std::string> shared_events;
    for (const auto &usage : usages_) {
      if (!IsConcreteCore(usage.core))
        shared_events.insert(usage.hard_event);
    }
    for (const auto &usage : usages_) {
      CoreMask core = split_core_banks && !shared_events.count(usage.hard_event)
                          ? usage.core
                          : kCoreUnassigned;
      FlagNamespace name_space{usage.hard_event, core};
      call_ranges[usage.call].push_back({usage.bounds, name_space});
      auto [it, inserted] = ranges.try_emplace(name_space);
      if (inserted)
        order.push_back(name_space);
      it->second.push_back(usage.bounds);
    }
  }

  void VisitStmt_(const SBlockNode *op) final {
    CoreMask saved = current_core_;
    CoreMask block_core = op->name_hint == "CUBE"     ? kCoreCube
                          : op->name_hint == "VECTOR" ? kCoreVector
                                                      : kCoreUnassigned;
    if (block_core != kCoreUnassigned) {
      current_core_ = saved == kCoreUnassigned || saved == block_core
                          ? block_core
                          : kCoreBroadcast;
    }
    // Ordinary nested blocks inherit ownership. Conflicting core wrappers
    // remain ambiguous instead of being treated as a third independent bank.
    ConstrVisitor::VisitStmt_(op);
    current_core_ = saved;
  }

  void VisitExpr_(const CallNode *op) final {
    if (std::optional<std::string> he = GetFlagHardEvent(op)) {
      EventIdBounds range = EventIdRange(GetConstrSet(), *he, op->args[1]);
      usages_.push_back({op, *he, current_core_, range});
    } else {
      ReserveGemmL1Bufs(op);
    }
    ConstrVisitor::VisitExpr_(op);
  }

private:
  struct FlagUsage {
    const CallNode *call;
    std::string hard_event;
    CoreMask core;
    EventIdBounds bounds;
  };
  std::vector<FlagUsage> usages_;
  CoreMask current_core_{kCoreUnassigned};

  // gemm_l1 templates internally occupy buf_ids [buf_offset, buf_offset+1].
  void ReserveGemmL1Bufs(const CallNode *op) {
    int arg_idx;
    if (op->op.same_as(ascend_gemm_l1())) {
      arg_idx = kGemmL1BufOffsetArg;
    } else if (op->op.same_as(ascend_blockscaled_gemm_l1())) {
      arg_idx = kBlockscaledGemmL1BufOffsetArg;
    } else {
      return;
    }
    ICHECK_LT(static_cast<size_t>(arg_idx), op->args.size())
        << "RewriteFlagToBuf: gemm_l1 call missing buf_offset arg";
    const auto *off = op->args[arg_idx].as<IntImmNode>();
    ICHECK(off) << "RewriteFlagToBuf: gemm_l1 buf_offset must be a constant";
    reserved_hwm = std::max(reserved_hwm, off->value + kGemmL1BufSlots);
  }
};

// Coalesce raw ranges into sync-point blocks: merge only overlapping/contained
// ranges (a block absorbs its point-priming ids); adjacent-disjoint ranges
// (e.g. [0,3] and [4,7]) stay as separate sync points.
std::vector<Block>
CoalesceBlocks(std::vector<std::pair<int64_t, int64_t>> ranges) {
  std::sort(ranges.begin(), ranges.end());
  std::vector<Block> blocks;
  for (const auto &r : ranges) {
    if (!blocks.empty() && r.first <= blocks.back().hi) {
      blocks.back().hi = std::max(blocks.back().hi, r.second);
    } else {
      blocks.push_back(Block{r.first, r.second});
    }
  }
  return blocks;
}

// 0/1 knapsack (capacity kFlagSlotsPerPair): pick the subset of blocks whose
// total size is maximal but <= capacity. Returns a keep-flag per block.
std::vector<bool> KnapsackKeep(const std::vector<Block> &blocks) {
  const int cap = kFlagSlotsPerPair;
  int n = static_cast<int>(blocks.size());
  // parent[w] = index of the block added to first reach subset-sum w (-1 = the
  // empty base); prev_sum[w] = the sum before adding it.
  std::vector<int> parent(cap + 1, -2);
  std::vector<int> prev_sum(cap + 1, -1);
  parent[0] = -1;
  for (int i = 0; i < n; ++i) {
    int wi = static_cast<int>(blocks[i].size());
    if (wi > cap)
      continue; // a block larger than the flag budget can never be kept.
    for (int w = cap; w >= wi; --w) {
      if (parent[w] == -2 && parent[w - wi] != -2) {
        parent[w] = i;
        prev_sum[w] = w - wi;
      }
    }
  }
  int best_w = 0;
  for (int w = cap; w >= 0; --w) {
    if (parent[w] != -2) {
      best_w = w;
      break;
    }
  }
  std::vector<bool> kept(n, false);
  for (int w = best_w; w > 0;) {
    int i = parent[w];
    kept[i] = true;
    w = prev_sum[w];
  }
  return kept;
}

// Phase B: rewrite each flag call per its block assignment.
class FlagToBufRewriter : public StmtExprMutator {
public:
  explicit FlagToBufRewriter(FlagAssignments assign, FlagCallRanges call_ranges)
      : assign_(std::move(assign)), call_ranges_(std::move(call_ranges)) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr)
      return StmtExprMutator::VisitStmt_(op);
    bool is_set = call->op.same_as(ascend_set_flag());
    bool is_wait = call->op.same_as(ascend_wait_flag());
    if (!is_set && !is_wait)
      return StmtExprMutator::VisitStmt_(op);

    const std::string &he = Downcast<StringImm>(call->args[0])->value;
    PrimExpr event_id = call->args[1];
    auto range_it = call_ranges_.find(call);
    ICHECK(range_it != call_ranges_.end())
        << "RewriteFlagToBuf: missing collected range for flag call";
    size_t &range_index = call_range_indices_[call];
    ICHECK_LT(range_index, range_it->second.size())
        << "RewriteFlagToBuf: exhausted collected ranges for flag call";
    const FlagCallRange &usage = range_it->second[range_index++];
    auto it = assign_.find(usage.name_space);
    // Consume every occurrence, including unchanged ones: the same CallNode
    // can be shared by different core regions or lexical loop contexts.
    if (it == assign_.end())
      return StmtExprMutator::VisitStmt_(op);
    const EventIdBounds &range = usage.bounds;
    const BlockAssign *b = nullptr;
    for (const auto &ba : it->second) {
      if (ba.lo <= range.first && range.second <= ba.hi) {
        b = &ba;
        break;
      }
    }
    ICHECK(b != nullptr) << "RewriteFlagToBuf: event_id range [" << range.first
                         << "," << range.second
                         << "] not covered by any block for '" << he << "'";

    PrimExpr new_id = b->delta == 0
                          ? event_id
                          : event_id + IntImm(event_id.dtype(), b->delta);

    if (b->kept) {
      // Stay as a flag, only renumbered into [0,8).
      Op op_id = is_set ? ascend_set_flag() : ascend_wait_flag();
      return Evaluate(Call(DataType::Handle(), op_id, {StringImm(he), new_id}));
    }
    // Spill to the mutex pool as a back-to-back asc_lock/asc_unlock pair.
    auto pipes = SplitHardEvent(he);
    std::string pipe = is_set ? pipes.first : pipes.second;
    int mode = is_set ? 1 : 0;
    return SeqStmt({MakeBufOp(ascend_get_buf(), pipe, new_id, mode),
                    MakeBufOp(ascend_rls_buf(), pipe, new_id, mode)});
  }

private:
  static Stmt MakeBufOp(const Op &op, const std::string &pipe,
                        const PrimExpr &buf_id, int mode) {
    return Evaluate(
        Call(DataType::Handle(), op,
             {StringImm(pipe), buf_id, IntImm(DataType::Int(32), mode)}));
  }

  FlagAssignments assign_;
  FlagCallRanges call_ranges_;
  std::unordered_map<const CallNode *, size_t> call_range_indices_;
};

PrimFunc RewriteFlagToBufFn(PrimFunc f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  bool split_core_banks =
      target.has_value() && GetAscendTargetInfo(target.value()).architecture ==
                                AscendArchitecture::kDav2201;
  FlagUsageCollector collector;
  collector(f->body);
  collector.FinalizeNamespaces(split_core_banks);
  if (collector.order.empty())
    return f;

  // CANN 9.1 c_api/sync/sync.h declares ascMutexExecuteMode and
  // asc_lock/asc_unlock only under __NPU_ARCH__ == 3510. Sparse compaction is
  // legal on both targets, but dav-2201 cannot lower a true flag overflow to
  // mutexes. Unbound single-pass inputs retain the upstream dav-3510 default.
  bool supports_mutex_spill =
      !target.has_value() || GetAscendTargetInfo(target.value()).architecture ==
                                 AscendArchitecture::kDav3510;

  FlagAssignments assign;
  // Start spilled buf_id allocation past the slots gemm_l1 already reserves.
  int mutex_running = static_cast<int>(collector.reserved_hwm);
  bool has_spill = false;
  std::ostringstream layout;
  for (const FlagNamespace &name_space : collector.order) {
    const std::string &he = name_space.first;
    std::vector<Block> blocks = CoalesceBlocks(collector.ranges[name_space]);
    int64_t total = 0;
    bool already_legal = true;
    for (const auto &b : blocks) {
      total += b.size();
      already_legal &= b.hi < kFlagSlotsPerPair;
    }
    if (total <= kFlagSlotsPerPair && already_legal)
      continue; // Already fits in the physical flag namespace.

    if (total > kFlagSlotsPerPair && !supports_mutex_spill) {
      std::ostringstream ranges;
      for (size_t i = 0; i < blocks.size(); ++i) {
        if (i != 0)
          ranges << ", ";
        ranges << "[" << blocks[i].lo << ".." << blocks[i].hi << "]";
      }
      TVM_FFI_THROW(ValueError)
          << "RewriteFlagToBuf: dav-2201 hard_event '" << he << "' requires "
          << total << " local event slots from ranges " << ranges.str()
          << (name_space.second == kCoreCube ? " in CUBE bank"
              : name_space.second == kCoreVector
                  ? " in VECTOR bank"
                  : " in conservative core union")
          << "; limit is " << kFlagSlotsPerPair
          << ". This is the allocated layout, not a minimum slot requirement"
          << ". Mutex spill requires dav-3510 asc_lock/asc_unlock support.";
    }

    // A sparse out-of-range layout that still fits in 8 slots only needs
    // compaction. Use knapsack selection only when capacity is truly exceeded.
    std::vector<bool> kept = total <= kFlagSlotsPerPair
                                 ? std::vector<bool>(blocks.size(), true)
                                 : KnapsackKeep(blocks);
    std::vector<BlockAssign> he_assign;
    int flag_running = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
      const Block &b = blocks[i];
      if (kept[i]) {
        ICHECK_LE(static_cast<int64_t>(flag_running) + b.size(),
                  static_cast<int64_t>(kFlagSlotsPerPair))
            << "RewriteFlagToBuf: compacted flag range exceeds [0,8) for '"
            << he << "'";
        he_assign.push_back(BlockAssign{b.lo, b.hi, true, flag_running - b.lo});
        flag_running += static_cast<int>(b.size());
      } else {
        has_spill = true;
        he_assign.push_back(
            BlockAssign{b.lo, b.hi, false, mutex_running - b.lo});
        layout << "\n  " << he << ": event_id[" << b.lo << ".." << b.hi
               << "] -> buf_id[" << mutex_running << ".."
               << (mutex_running + b.size() - 1) << "]";
        mutex_running += static_cast<int>(b.size());
      }
    }
    assign[name_space] = std::move(he_assign);
  }

  if (assign.empty())
    return f; // Nothing needs compaction or spilling.
  if (has_spill) {
    ICHECK_LE(mutex_running, kMaxBufId)
        << "RewriteFlagToBuf: total spilled mutex slots (" << mutex_running
        << ") exceed the shared pool of " << kMaxBufId << "." << layout.str();
  }

  FlagToBufRewriter rewriter(std::move(assign),
                             std::move(collector.call_ranges));
  f.CopyOnWrite()->body = rewriter(f->body);
  return f;
}

} // namespace

using namespace tirx::transform;

tvm::transform::Pass RewriteFlagToBuf() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return RewriteFlagToBufFn(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RewriteFlagToBuf", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteFlagToBuf", RewriteFlagToBuf);
}

} // namespace tl
} // namespace tvm
