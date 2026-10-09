/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file infer_buffer_aliases.cc
 * \brief Infer pairwise buffer-reuse compatibility for manual schedules.
 */
#include "buffer_alias.h"
#include "merge_ub_common.h"
#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/op/builtin.h"
#include "ascend/target_utils.h"
#include "ascend_pipe.h"
#include "core_mask.h"
#include "op/builtin.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using namespace merge_ub;

// Sync-Aware Linearizer

enum class EventKind : uint8_t {
  kBufferAccess,  // BufferLoad or BufferStore
  kSetFlag,       // ascend_set_flag
  kWaitFlag,      // ascend_wait_flag
  kPipeBarrier,   // ascend_pipe_barrier
  kCrossCoreSet,  // ascend_cross_core_set_flag
  kCrossCoreWait, // ascend_cross_core_wait_flag
  kForBegin,      // entry of a For loop body
  kForEnd,        // exit of a For loop body
  kIfBegin,       // entry of IfThenElse then-branch
  kIfEnd,         // exit of IfThenElse
  kElseBegin,     // entry of IfThenElse else-branch
  kOpaqueBegin,   // a loop without a finite execution domain
  kOpaqueEnd,
};

// The hardware pipe an event executes on. Textual adjacency only implies a
// genuine hardware happens-before within one pipe: different pipes run
// concurrently and are ordered solely by explicit flag handshakes
// (AddSyncEdges). Program-order wiring (rules 1/2/3 in
// HappensBeforeGraph::AddProgramOrderEdges) must therefore be scoped per
// pipe; wiring across pipes fabricates ordering the hardware never
// provides (e.g. a Fixpipe wait followed by an MTE1 store — the wait does
// not block MTE1 at all).
struct LinearEvent {
  EventKind kind;
  int scope_level; // nesting depth

  // Buffer access
  std::vector<const VarNode *> touched;
  std::vector<bool> is_write; // parallel to touched

  // Sync primitives
  std::string hard_event;
  PrimExpr event_id_expr; // raw event_id expression (may contain loop vars)
  std::string pipe_name;  // for pipe_barrier
  int mode_id = -1;       // unknown mode never supplies an ordering proof
  CoreMask core_mask = kCoreUnassigned;
  ResourcePipe exec_pipe = ResourcePipe::kUnknown;

  // Scope markers
  const ForNode *for_node = nullptr;
  const IfThenElseNode *if_node = nullptr;
  bool finite_loop_control = true;

  // Immutable, occurrence-local lexical facts. Names are never bindings.
  Map<Var, PrimExpr> bindings;
  std::vector<PrimExpr> guards;
  std::vector<std::pair<Var, PrimExpr>> lane_bindings;

  // Alloc info
  struct AllocInfo {
    const AllocBufferNode *alloc = nullptr;
    size_t scope_level = 0;
  };
  bool IsBufferAccess() const { return kind == EventKind::kBufferAccess; }
  bool IsSync() const {
    return kind == EventKind::kSetFlag || kind == EventKind::kWaitFlag ||
           kind == EventKind::kPipeBarrier ||
           kind == EventKind::kCrossCoreSet ||
           kind == EventKind::kCrossCoreWait;
  }
};

class SyncAwareLinearizer : public StmtExprVisitor {
public:
  explicit SyncAwareLinearizer(const std::string &target_scope)
      : target_scope_(target_scope) {}

  std::vector<LinearEvent> linear_events;
  std::unordered_map<const VarNode *, LinearEvent::AllocInfo> alloc_info;
  // Pipe-relevant op names that the shared classifier could not map.
  std::unordered_set<std::string> unknown_instr_ops;

private:
  bool IsTargetMemory(const Var &var) {
    if (!target_scope_.empty())
      return IsScopeMemory(var, target_scope_);
    return IsDynamicSharedMemory(var);
  }

  void VisitControlExpr(const PrimExpr &value) {
    // Control evaluation is scalar even when lexically inside VECTOR. Native
    // calls still select their own argument pipes in VisitExpr_(CallNode).
    pipe_stack_.push_back(ResourcePipe::kScalar);
    VisitExpr(value);
    pipe_stack_.pop_back();
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    size_t level = scope_stack_.size();
    const VarNode *buf = op->buffer->data.get();
    if (IsTargetMemory(op->buffer->data)) {
      LinearEvent::AllocInfo info;
      info.alloc = op;
      info.scope_level = level;
      alloc_info[buf] = info;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    VisitControlExpr(op->value);
    bool immutable = SideEffect(op->value) <= CallEffectKind::kPure;
    PostOrderVisit(op->value, [&](const ObjectRef &node) {
      immutable &= !node.as<BufferLoadNode>() && !node.as<ProducerLoadNode>();
    });
    PrimExpr value = Substitute(op->value, bindings_);
    bindings_.erase(op->var);
    if (immutable)
      bindings_.Set(op->var, value);
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    auto saved = bindings_;
    StmtExprVisitor::VisitStmt_(op);
    bindings_ = std::move(saved);
  }

  void VisitStmt_(const ForNode *op) final {
    // Headers execute outside the repeated body.
    VisitControlExpr(op->min);
    VisitControlExpr(op->extent);
    if (op->step.defined())
      VisitControlExpr(op->step.value());
    LinearEvent begin;
    begin.kind = EventKind::kForBegin;
    begin.for_node = op;
    PostOrderVisit(op->body, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>())
        if (call->op.same_as(tl::loop_break()) ||
            call->op.same_as(builtin::break_loop()) ||
            call->op.same_as(builtin::continue_loop()))
          begin.finite_loop_control = false;
    });
    Record(begin);
    auto saved = bindings_;
    bindings_.erase(op->loop_var);
    VisitStmt(op->body);
    bindings_ = std::move(saved);
    LinearEvent end;
    end.kind = EventKind::kForEnd;
    end.for_node = op;
    Record(end);
  }

  void VisitStmt_(const WhileNode *op) final {
    LinearEvent begin;
    begin.kind = EventKind::kOpaqueBegin;
    Record(begin);
    // The condition is re-evaluated at every loop back edge. Its accesses
    // belong to the whole-loop may-live summary, outside its own assumption.
    VisitControlExpr(op->condition);
    auto saved = bindings_;
    VisitStmt(op->body);
    bindings_ = std::move(saved);
    LinearEvent end;
    end.kind = EventKind::kOpaqueEnd;
    Record(end);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitControlExpr(op->condition);
    LinearEvent if_begin;
    if_begin.kind = EventKind::kIfBegin;
    if_begin.scope_level = static_cast<int>(scope_stack_.size());
    if_begin.if_node = op;
    Record(if_begin);

    auto saved = bindings_;
    guards_.push_back(op->condition);
    VisitStmt(op->then_case);
    guards_.pop_back();
    bindings_ = saved;

    if (op->else_case.defined()) {
      LinearEvent else_begin;
      else_begin.kind = EventKind::kElseBegin;
      else_begin.if_node = op;
      Record(else_begin);
      guards_.push_back(!op->condition);
      VisitStmt(op->else_case.value());
      guards_.pop_back();
      bindings_ = saved;
    }

    LinearEvent if_end;
    if_end.kind = EventKind::kIfEnd;
    if_end.scope_level = static_cast<int>(scope_stack_.size());
    if_end.if_node = op;
    Record(if_end);
  }

  void VisitStmt_(const SBlockNode *op) final {
    auto saved_bindings = bindings_;
    CoreMask saved = current_core_mask_;
    if (op->name_hint == "VECTOR")
      current_core_mask_ = kCoreVector;
    else if (op->name_hint == "CUBE")
      current_core_mask_ = kCoreCube;
    // Scope blocks provide the pipe for bare BufferStore/BufferLoad accesses
    // that are not nested in an instruction call.
    ResourcePipe block_pipe = GetAscendBlockPipe(op->name_hint);
    bool has_block_pipe = block_pipe != ResourcePipe::kUnknown;
    if (has_block_pipe)
      pipe_stack_.push_back(block_pipe);
    StmtExprVisitor::VisitStmt_(op);
    if (has_block_pipe)
      pipe_stack_.pop_back();
    current_core_mask_ = saved;
    bindings_ = std::move(saved_bindings);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    for (const PrimExpr &value : op->iter_values)
      VisitControlExpr(value);
    VisitControlExpr(op->predicate);
    guards_.push_back(op->predicate);
    VisitStmt(op->block);
    guards_.pop_back();
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    auto saved = bindings_;
    const auto *iv = op->node.as<IterVarNode>();
    bool lane = (op->attr_key == tirx::attr::thread_extent ||
                 op->attr_key == s_tir::attr::virtual_thread) &&
                iv && iv->thread_tag == "cthread";
    if (lane)
      lane_bindings_.emplace_back(iv->var, op->value);
    bool scope = op->attr_key == tirx::attr::thread_extent ||
                 op->attr_key == tirx::attr::extern_scope ||
                 op->attr_key == s_tir::attr::virtual_thread;
    if (scope)
      scope_stack_.push_back(op);
    VisitStmt(op->body);
    if (scope)
      scope_stack_.pop_back();
    if (lane)
      lane_bindings_.pop_back();
    bindings_ = std::move(saved);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    const VarNode *buf = op->buffer->data.get();
    if (alloc_info.count(buf)) {
      LinearEvent e;
      e.kind = EventKind::kBufferAccess;
      e.scope_level = static_cast<int>(scope_stack_.size());
      e.touched.push_back(buf);
      e.is_write.push_back(true);
      e.core_mask = current_core_mask_;
      e.exec_pipe = CurrentExecPipe();
      Record(e);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    const VarNode *buf = op->buffer->data.get();
    if (alloc_info.count(buf)) {
      LinearEvent e;
      e.kind = EventKind::kBufferAccess;
      e.scope_level = static_cast<int>(scope_stack_.size());
      e.touched.push_back(buf);
      e.is_write.push_back(false);
      e.core_mask = current_core_mask_;
      e.exec_pipe = CurrentExecPipe();
      Record(e);
    }
  }

  void VisitExpr_(const VarNode *op) final {
    // Direct variable references count as reads for liveness.  Buffer-var
    // arguments of tvm_access_ptr are handled by the CallNode override below
    // (which knows the read/write mask); suppress the generic read here so we
    // do not mislabel every access_ptr operand as a read.
    if (suppress_var_read_.count(op))
      return;
    if (alloc_info.count(op)) {
      LinearEvent e;
      e.kind = EventKind::kBufferAccess;
      e.scope_level = static_cast<int>(scope_stack_.size());
      e.touched.push_back(op);
      e.is_write.push_back(false);
      e.core_mask = current_core_mask_;
      e.exec_pipe = CurrentExecPipe();
      Record(e);
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    // Check if this Evaluate wraps a sync Call.
    if (auto *call = op->value.as<CallNode>()) {
      if (TryRecordSync(call))
        return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (TryRecordSync(op))
      return;
    Call call = GetRef<Call>(op);
    uint16_t pipe_mask = GetAscendCallPipeMask(call);
    // Composite instructions expose a pipe for each operand. In particular,
    // gemm_l1 reads L1 inputs on MTE1 and writes its L0C result on Cube.
    if (pipe_mask != kAllResourcePipes && pipe_mask != 0 &&
        !IsSingleResourcePipe(pipe_mask)) {
      for (size_t arg_idx = 0; arg_idx < op->args.size(); ++arg_idx) {
        ResourcePipe operand_pipe = GetAscendCallArgumentPipe(call, arg_idx);
        pipe_stack_.push_back(operand_pipe);
        VisitExpr(op->args[arg_idx]);
        pipe_stack_.pop_back();
      }
      return;
    }

    ResourcePipe inst_pipe = GetAscendCallArgumentPipe(call, 0);
    bool pushed_pipe = false;
    if (inst_pipe != ResourcePipe::kUnknown) {
      pipe_stack_.push_back(inst_pipe);
      pushed_pipe = true;
    } else if (!op->op.same_as(builtin::tvm_access_ptr()) &&
               IsAscendPipeRelevantCall(call)) {
      unknown_instr_ops.insert(call->op.as<Op>().value()->name);
    }
    // tvm_access_ptr(type, buffer_var, offset, extent, rw_mask): record the
    // access with the correct read/write flag from rw_mask (1=read, 2=write,
    // 3=read+write) instead of letting the bare buffer_var count as a read.
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() >= 5) {
      if (const auto *buf = op->args[1].as<VarNode>()) {
        if (alloc_info.count(buf)) {
          int64_t rw = 0;
          if (const auto *imm = op->args[4].as<IntImmNode>())
            rw = imm->value;
          bool is_read = (rw & 1) != 0;
          bool is_write = (rw & 2) != 0;
          // A pure access_ptr with an unknown/zero mask is treated as a read
          // (conservative: keeps the buffer live) rather than being dropped.
          if (!is_read && !is_write)
            is_read = true;
          LinearEvent e;
          e.kind = EventKind::kBufferAccess;
          e.scope_level = static_cast<int>(scope_stack_.size());
          e.touched.push_back(buf);
          e.is_write.push_back(is_write);
          e.core_mask = current_core_mask_;
          e.exec_pipe = CurrentExecPipe();
          Record(e);
          // Suppress the generic VarNode read for this buffer var while we
          // visit the rest of the access_ptr sub-expression.
          suppress_var_read_.insert(buf);
          for (const PrimExpr &arg : op->args)
            VisitExpr(arg);
          suppress_var_read_.erase(buf);
          return;
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
    if (pushed_pipe) {
      pipe_stack_.pop_back();
    }
  }

  bool TryRecordSync(const CallNode *call) {
    bool sync = call->op.same_as(ascend_set_flag()) ||
                call->op.same_as(ascend_wait_flag()) ||
                call->op.same_as(ascend_pipe_barrier()) ||
                call->op.same_as(ascend_cross_core_set_flag()) ||
                call->op.same_as(ascend_cross_core_wait_flag());
    if (!sync)
      return false;
    // An unresolved ID cannot supply an edge, but computing it may still read
    // storage whose lifetime must be included in every reuse proof.
    for (const PrimExpr &arg : call->args)
      VisitControlExpr(arg);
    LinearEvent e;
    e.scope_level = static_cast<int>(scope_stack_.size());

    if (call->op.same_as(ascend_set_flag())) {
      e.kind = EventKind::kSetFlag;
      ICHECK_GE(call->args.size(), 2U);
      if (auto *str_imm = call->args[0].as<StringImmNode>()) {
        e.hard_event = str_imm->value;
        e.exec_pipe = GetFlagEndpointPipe(e.hard_event, FlagEndpoint::kSet);
      }
      e.event_id_expr = call->args[1];
      e.core_mask = current_core_mask_;
      Record(e);
      return true;
    }
    if (call->op.same_as(ascend_wait_flag())) {
      e.kind = EventKind::kWaitFlag;
      ICHECK_GE(call->args.size(), 2U);
      if (auto *str_imm = call->args[0].as<StringImmNode>()) {
        e.hard_event = str_imm->value;
        e.exec_pipe = GetFlagEndpointPipe(e.hard_event, FlagEndpoint::kWait);
      }
      e.event_id_expr = call->args[1];
      e.core_mask = current_core_mask_;
      Record(e);
      return true;
    }
    if (call->op.same_as(ascend_pipe_barrier())) {
      e.kind = EventKind::kPipeBarrier;
      ICHECK_GE(call->args.size(), 1U);
      if (auto *str_imm = call->args[0].as<StringImmNode>()) {
        e.hard_event = str_imm->value; // store pipe name in hard_event
        e.exec_pipe = ParseAscendPipe(e.hard_event);
      }
      e.core_mask = current_core_mask_;
      Record(e);
      return true;
    }
    if (call->op.same_as(ascend_cross_core_set_flag())) {
      e.kind = EventKind::kCrossCoreSet;
      ICHECK_GE(call->args.size(), 3U);
      if (auto *int_imm = call->args[0].as<IntImmNode>())
        if (int_imm->value >= 0 &&
            int_imm->value <= std::numeric_limits<int>::max())
          e.mode_id = static_cast<int>(int_imm->value);
      if (auto *str_imm = call->args[1].as<StringImmNode>()) {
        e.hard_event = str_imm->value; // store pipe name
        e.exec_pipe = ParseAscendPipe(e.hard_event);
      }
      e.event_id_expr = call->args[2];
      e.core_mask = current_core_mask_;
      Record(e);
      return true;
    }
    if (call->op.same_as(ascend_cross_core_wait_flag())) {
      e.kind = EventKind::kCrossCoreWait;
      ICHECK_GE(call->args.size(), 3U);
      if (auto *int_imm = call->args[0].as<IntImmNode>())
        if (int_imm->value >= 0 &&
            int_imm->value <= std::numeric_limits<int>::max())
          e.mode_id = static_cast<int>(int_imm->value);
      if (auto *str_imm = call->args[1].as<StringImmNode>()) {
        e.hard_event = str_imm->value;
        e.exec_pipe = ParseAscendPipe(e.hard_event);
      }
      e.event_id_expr = call->args[2];
      e.core_mask = current_core_mask_;
      Record(e);
      return true;
    }
    return false;
  }

  void Record(LinearEvent event) {
    event.bindings = bindings_;
    event.guards = guards_;
    event.lane_bindings = lane_bindings_;
    event.core_mask = current_core_mask_;
    linear_events.push_back(std::move(event));
  }

  std::string target_scope_;
  CoreMask current_core_mask_ = kCoreUnassigned;
  Map<Var, PrimExpr> bindings_;
  std::vector<PrimExpr> guards_;
  std::vector<std::pair<Var, PrimExpr>> lane_bindings_;
  std::vector<ResourcePipe> pipe_stack_;
  ResourcePipe CurrentExecPipe() const {
    return pipe_stack_.empty() ? ResourcePipe::kUnknown : pipe_stack_.back();
  }
  std::unordered_set<const VarNode *> suppress_var_read_;
  std::vector<const Object *> scope_stack_;
};

// A physical owner has its own local event bank and instruction FIFOs. AIV
// lanes share L1/L0, but their UB allocations are independent.
using ExecutionOwner = std::pair<CoreMask, int>;

struct EventInstance {
  int node_id = 0;
  int linear_idx = 0;
  int issue_order = 0;
  EventKind kind;
  std::vector<const VarNode *> touched;
  std::vector<bool> is_write;
  std::string hard_event;
  PrimExpr event_id_expr;
  std::optional<int64_t> event_id;
  int mode_id = -1;
  CoreMask core_mask = kCoreUnassigned;
  int lane = 0;
  ResourcePipe exec_pipe = ResourcePipe::kUnknown;
  Map<Var, PrimExpr> bindings;
  std::vector<std::pair<const ForNode *, int64_t>> iterations;
  std::vector<int> loop_ids;
  bool definite = true;
  bool summary = false;
  ExecutionOwner Owner() const { return {core_mask, lane}; }
};

struct HappensBeforeGraph {
  static constexpr size_t kMaxInstances = 8000;
  std::vector<EventInstance> instances;
  std::vector<std::vector<bool>> reachable;
  int N = 0;
  bool within_budget = true;
  bool complete_owner_domain = true;

  void Build(const std::vector<LinearEvent> &events,
             std::optional<AscendArchitecture> architecture) {
    architecture_ = architecture;
    events_ = &events;
    matching_end_.assign(events.size(), -1);
    std::vector<int> nesting;
    std::set<CoreMask> cores;
    for (size_t i = 0; i < events.size(); ++i) {
      const auto &event = events[i];
      if (event.kind == EventKind::kForBegin ||
          event.kind == EventKind::kOpaqueBegin) {
        nesting.push_back(static_cast<int>(i));
      } else if (event.kind == EventKind::kForEnd ||
                 event.kind == EventKind::kOpaqueEnd) {
        ICHECK(!nesting.empty());
        matching_end_[nesting.back()] = static_cast<int>(i);
        nesting.pop_back();
      }
      if (event.IsBufferAccess() || event.IsSync())
        cores.insert(event.core_mask);
      for (const auto &[var, extent] : event.lane_bindings) {
        arith::Analyzer analyzer;
        PrimExpr count = analyzer.Simplify(Substitute(extent, event.bindings));
        const auto *imm = count.as<IntImmNode>();
        if (!imm || imm->value < 1 || imm->value > 2) {
          complete_participants_ = false;
          complete_owner_domain = false;
        } else {
          num_aiv_ = std::max(num_aiv_, static_cast<int>(imm->value));
          has_lane_binding_ = true;
        }
      }
    }
    ICHECK(nesting.empty());
    mixed_owners_ = cores.count(kCoreUnassigned) && cores.size() > 1;
    // An unknown physical participant domain cannot be represented by a
    // fabricated lane-zero execution: it could erase actual guarded accesses.
    if (!complete_owner_domain) {
      events_ = nullptr;
      return;
    }
    for (CoreMask core : cores) {
      int count = core != kCoreCube ? num_aiv_ : 1;
      for (int lane = 0; lane < count; ++lane) {
        owner_ = {core, lane};
        issue_order_ = 0;
        if (!Expand(0, static_cast<int>(events.size()), {}, {}, {})) {
          // Even the finite may-access summary must fit the global bound.
          // No reachability proof is published when the input itself exceeds
          // it.
          within_budget = false;
          instances.clear();
          return;
        }
      }
    }
    N = static_cast<int>(instances.size());
    reachable.assign(N, std::vector<bool>(N, false));
    for (int i = 0; i < N; ++i)
      reachable[i][i] = true;
    AddProgramOrderEdges();
    AddSyncEdges();
    AddPipeBarrierEdges();
    ComputeTransitiveClosure();
    events_ = nullptr;
  }

  void GetAllBufferAccessNodes(const VarNode *var,
                               std::vector<int> &ids) const {
    for (int i = 0; i < N; ++i)
      if (instances[i].kind == EventKind::kBufferAccess &&
          std::find(instances[i].touched.begin(), instances[i].touched.end(),
                    var) != instances[i].touched.end())
        ids.push_back(i);
  }

private:
  const std::vector<LinearEvent> *events_ = nullptr;
  std::vector<int> matching_end_;
  std::optional<AscendArchitecture> architecture_;
  ExecutionOwner owner_;
  int issue_order_ = 0;
  int num_aiv_ = 1;
  bool has_lane_binding_ = false;
  bool complete_participants_ = true;
  bool mixed_owners_ = false;
  std::set<std::string> poisoned_namespaces_;
  std::set<std::string> ambiguous_local_events_;

  Map<Var, PrimExpr> Bindings(const LinearEvent &event,
                              const Map<Var, PrimExpr> &iterations) const {
    Map<Var, PrimExpr> result = iterations;
    for (const auto &[var, extent] : event.lane_bindings)
      result.Set(var, IntImm(var.dtype(), owner_.second));
    for (const auto &[var, value] : event.bindings)
      result.Set(var, Substitute(value, result));
    return result;
  }

  PrimExpr Resolve(const PrimExpr &value, const LinearEvent &event,
                   const Map<Var, PrimExpr> &iterations) const {
    // Lexical Bind RHSs were already expanded in source order. Substitute
    // those first, then the actual enclosing For/cthread values by identity.
    PrimExpr result = Substitute(value, event.bindings);
    arith::Analyzer analyzer;
    return analyzer.Simplify(Substitute(result, Bindings(event, iterations)));
  }

  std::optional<int64_t> Constant(const PrimExpr &value,
                                  const LinearEvent &event,
                                  const Map<Var, PrimExpr> &iterations) const {
    PrimExpr result = Resolve(value, event, iterations);
    if (const auto *imm = result.as<IntImmNode>())
      return imm->value;
    return std::nullopt;
  }

  // False excludes an occurrence. Unknown retains a may-access, but never a
  // synchronization witness. A branch cannot lend its assumption to itself.
  int Active(const LinearEvent &event,
             const Map<Var, PrimExpr> &iterations) const {
    bool unknown = false;
    for (const auto &[var, extent] : event.lane_bindings) {
      auto count = Constant(extent, event, iterations);
      if (!count || *count < 1 || *count > 2)
        unknown = true;
      else if (owner_.first != kCoreCube && owner_.second >= *count)
        return 0;
    }
    for (const PrimExpr &guard : event.guards) {
      auto value = Constant(guard, event, iterations);
      if (value && !*value)
        return 0;
      unknown |= !value.has_value();
    }
    return unknown ? -1 : 1;
  }

  std::string Namespace(const EventInstance &event) const {
    if (event.kind == EventKind::kCrossCoreSet ||
        event.kind == EventKind::kCrossCoreWait)
      return "X"; // Modes share the same physical counter namespace.
    return "L:" + std::to_string(event.core_mask) + ":" +
           std::to_string(event.lane) + ":" + event.hard_event;
  }

  bool Append(int index, const Map<Var, PrimExpr> &bindings,
              const std::vector<std::pair<const ForNode *, int64_t>> &tuple,
              const std::vector<int> &loop_ids, bool summary) {
    const auto &event = (*events_)[index];
    if (event.core_mask != owner_.first ||
        (!event.IsBufferAccess() && !event.IsSync()))
      return true;
    int active = Active(event, bindings);
    if (!active)
      return true;
    EventInstance instance;
    instance.node_id = static_cast<int>(instances.size());
    instance.linear_idx = index;
    instance.issue_order = issue_order_++;
    instance.kind = event.kind;
    instance.touched = event.touched;
    instance.is_write = event.is_write;
    instance.hard_event = event.hard_event;
    instance.mode_id = event.mode_id;
    instance.core_mask = owner_.first;
    instance.lane = owner_.second;
    instance.exec_pipe = event.exec_pipe;
    instance.bindings = Bindings(event, bindings);
    instance.iterations = tuple;
    instance.loop_ids = loop_ids;
    instance.definite = active > 0 && !summary;
    instance.summary = summary;
    if (event.event_id_expr.defined()) {
      instance.event_id_expr = Resolve(event.event_id_expr, event, bindings);
      if (const auto *imm = instance.event_id_expr.as<IntImmNode>())
        instance.event_id = imm->value;
    }
    if (event.IsSync() && event.kind != EventKind::kPipeBarrier &&
        (!instance.definite || !instance.event_id))
      poisoned_namespaces_.insert(Namespace(instance));
    if ((event.kind == EventKind::kSetFlag ||
         event.kind == EventKind::kWaitFlag) &&
        instance.core_mask == kCoreUnassigned && mixed_owners_)
      // This endpoint can consume a concrete core's token. It is not a third
      // physical event bank, and cannot be omitted from that bank's matching.
      ambiguous_local_events_.insert(instance.hard_event);
    if ((event.kind == EventKind::kCrossCoreSet ||
         event.kind == EventKind::kCrossCoreWait) &&
        (architecture_ != AscendArchitecture::kDav2201 ||
         instance.mode_id != 2 || !has_lane_binding_ ||
         !complete_participants_ || num_aiv_ != 2 ||
         !IsConcreteCore(instance.core_mask)))
      // Unsupported topology/mode endpoints still occupy this physical
      // namespace. Dropping them must not re-pair other set/wait occurrences.
      poisoned_namespaces_.insert("X");
    // Internal barriers are not loop-wide completion; unknown branch barriers
    // similarly cannot order an occurrence that may not execute.
    if (event.IsSync() && !instance.definite)
      return true;
    if (instances.size() >= kMaxInstances)
      return false;
    instances.push_back(std::move(instance));
    return true;
  }

  bool Summarize(int first, int last, const Map<Var, PrimExpr> &bindings,
                 const std::vector<std::pair<const ForNode *, int64_t>> &tuple,
                 std::vector<int> loop_ids) {
    loop_ids.push_back(first - 1);
    for (int index = first; index < last; ++index)
      if (!Append(index, bindings, tuple, loop_ids, true))
        return false;
    return true;
  }

  bool Expand(int first, int last, const Map<Var, PrimExpr> &bindings,
              const std::vector<std::pair<const ForNode *, int64_t>> &tuple,
              const std::vector<int> &loop_ids) {
    for (int index = first; index < last; ++index) {
      const auto &event = (*events_)[index];
      if (event.kind != EventKind::kForBegin &&
          event.kind != EventKind::kOpaqueBegin) {
        if (!Append(index, bindings, tuple, loop_ids, false))
          return false;
        continue;
      }
      int end = matching_end_[index];
      ICHECK_GT(end, index);
      const auto *loop = event.for_node;
      auto start = loop ? Constant(loop->min, event, bindings) : std::nullopt;
      auto extent =
          loop ? Constant(loop->extent, event, bindings) : std::nullopt;
      auto step =
          loop ? Constant(loop->step.value_or(Integer(1)), event, bindings)
               : std::nullopt;
      // For's upper bound is min+extent; extent is not the trip count when
      // step differs from one. Use wide arithmetic before materializing values.
      int bits = loop ? loop->loop_var.dtype().bits() : 0;
      bool serial = loop && (loop->kind == ForKind::kSerial ||
                             loop->kind == ForKind::kUnrolled);
      bool finite = start && extent && step && *extent >= 0 && *step > 0 &&
                    bits > 0 && bits <= 64 && serial &&
                    event.finite_loop_control && Active(event, bindings) > 0;
      if (finite) {
        __int128 end_value = static_cast<__int128>(*start) + *extent;
        __int128 high = loop->loop_var.dtype().is_uint()
                            ? (static_cast<__int128>(1) << bits) - 1
                            : (static_cast<__int128>(1) << (bits - 1)) - 1;
        finite &= end_value <= high;
      }
      int64_t trips = finite ? *extent / *step + (*extent % *step != 0) : 0;
      size_t saved_size = instances.size();
      int saved_order = issue_order_;
      bool exact = finite && static_cast<uint64_t>(trips) <= kMaxInstances;
      if (exact) {
        for (int64_t iteration = 0; iteration < trips; ++iteration) {
          __int128 value = static_cast<__int128>(*start) +
                           static_cast<__int128>(iteration) * *step;
          __int128 low = loop->loop_var.dtype().is_uint()
                             ? 0
                             : -(static_cast<__int128>(1) << (bits - 1));
          __int128 high = loop->loop_var.dtype().is_uint()
                              ? (static_cast<__int128>(1) << bits) - 1
                              : (static_cast<__int128>(1) << (bits - 1)) - 1;
          if (bits > 64 || value < low || value > high) {
            exact = false;
            break;
          }
          auto nested = bindings;
          nested.Set(loop->loop_var, IntImm(loop->loop_var.dtype(),
                                            static_cast<int64_t>(value)));
          auto nested_tuple = tuple;
          nested_tuple.emplace_back(loop, iteration);
          auto nested_loops = loop_ids;
          if (trips > 1)
            nested_loops.push_back(index);
          if (!Expand(index + 1, end, nested, nested_tuple, nested_loops)) {
            exact = false;
            break;
          }
        }
      }
      if (!exact) {
        instances.resize(saved_size);
        issue_order_ = saved_order;
        if (!Summarize(index + 1, end, bindings, tuple, loop_ids))
          return false;
      }
      index = end;
    }
    return true;
  }

  bool UsableSync(const EventInstance &event) const {
    return event.definite && event.exec_pipe != ResourcePipe::kUnknown &&
           !((event.kind == EventKind::kSetFlag ||
              event.kind == EventKind::kWaitFlag) &&
             ambiguous_local_events_.count(event.hard_event)) &&
           poisoned_namespaces_.count(Namespace(event)) == 0;
  }

  void AddEdge(int from, int to) { reachable[from][to] = true; }

  void AddProgramOrderEdges() {
    // Keep the established release/acquire proof boundary, now in the actual
    // per-owner, per-pipe execution sequence across every enclosing loop.
    std::map<std::tuple<CoreMask, int, ResourcePipe>, std::vector<int>> tracks;
    for (int index = 0; index < N; ++index) {
      const auto &event = instances[index];
      if (event.exec_pipe != ResourcePipe::kUnknown &&
          (event.kind == EventKind::kBufferAccess || UsableSync(event)))
        tracks[{event.core_mask, event.lane, event.exec_pipe}].push_back(index);
    }
    auto release = [&](int index) {
      auto kind = instances[index].kind;
      return kind == EventKind::kSetFlag || kind == EventKind::kCrossCoreSet ||
             kind == EventKind::kPipeBarrier;
    };
    auto acquire = [&](int index) {
      auto kind = instances[index].kind;
      return kind == EventKind::kWaitFlag ||
             kind == EventKind::kCrossCoreWait ||
             kind == EventKind::kPipeBarrier;
    };
    for (auto &[key, track] : tracks) {
      int previous_sync = -1, previous_acquire = -1;
      for (int index : track) {
        if (instances[index].kind != EventKind::kBufferAccess) {
          if (previous_sync >= 0)
            AddEdge(previous_sync, index);
          previous_sync = index;
          if (acquire(index))
            previous_acquire = index;
        } else if (previous_acquire >= 0) {
          AddEdge(previous_acquire, index);
        }
      }
      int next_release = -1;
      for (auto it = track.rbegin(); it != track.rend(); ++it)
        if (release(*it))
          next_release = *it;
        else if (instances[*it].kind == EventKind::kBufferAccess &&
                 next_release >= 0)
          AddEdge(*it, next_release);
    }
    // On 2201 mode-2 wait_flag_dev(id) blocks subsequent instruction issue
    // on this participant, irrespective of its nominal pipe argument. Arrive
    // remains a completion release on its declared pipe only.
    if (architecture_ == AscendArchitecture::kDav2201)
      for (int wait = 0; wait < N; ++wait) {
        const auto &event = instances[wait];
        if (event.kind != EventKind::kCrossCoreWait || event.mode_id != 2 ||
            !UsableSync(event))
          continue;
        for (int next = 0; next < N; ++next)
          if (instances[next].Owner() == event.Owner() &&
              instances[next].issue_order > event.issue_order)
            AddEdge(wait, next);
      }
  }

  void AddSyncEdges() {
    using LocalKey = std::tuple<CoreMask, int, std::string, int64_t>;
    std::map<LocalKey, std::vector<int>> local;
    std::map<int64_t, std::vector<int>> collective;
    for (int index = 0; index < N; ++index) {
      const auto &event = instances[index];
      if (!UsableSync(event) || !event.event_id)
        continue;
      if (event.kind == EventKind::kSetFlag ||
          event.kind == EventKind::kWaitFlag) {
        local[{event.core_mask, event.lane, event.hard_event, *event.event_id}]
            .push_back(index);
      } else if (architecture_ == AscendArchitecture::kDav2201 &&
                 event.mode_id == 2 && has_lane_binding_ &&
                 complete_participants_ && num_aiv_ == 2) {
        collective[*event.event_id].push_back(index);
      }
    }
    for (const auto &[key, endpoints] : local) {
      std::vector<int> pending;
      for (int index : endpoints)
        if (instances[index].kind == EventKind::kSetFlag)
          pending.push_back(index);
        else if (!pending.empty()) {
          AddEdge(pending.front(), index);
          pending.erase(pending.begin());
        }
    }
    auto collect = [&](const std::vector<int> &endpoints, CoreMask core,
                       int lane, bool is_set) {
      std::vector<int> result;
      for (int index : endpoints) {
        const auto &event = instances[index];
        bool set = event.kind == EventKind::kCrossCoreSet;
        if (event.core_mask == core && event.lane == lane && set == is_set)
          result.push_back(index);
      }
      return result;
    };
    for (const auto &[id, endpoints] : collective) {
      auto cube_sets = collect(endpoints, kCoreCube, 0, true);
      auto cube_waits = collect(endpoints, kCoreCube, 0, false);
      std::vector<std::vector<int>> lane_sets, lane_waits;
      bool release_complete = true, ready_complete = true;
      for (int lane = 0; lane < num_aiv_; ++lane) {
        lane_sets.push_back(collect(endpoints, kCoreVector, lane, true));
        lane_waits.push_back(collect(endpoints, kCoreVector, lane, false));
        release_complete &= lane_sets.back().size() == cube_waits.size();
        ready_complete &= lane_waits.back().size() == cube_sets.size();
      }
      if (release_complete)
        for (size_t occurrence = 0; occurrence < cube_waits.size();
             ++occurrence)
          for (int lane = 0; lane < num_aiv_; ++lane)
            AddEdge(lane_sets[lane][occurrence], cube_waits[occurrence]);
      if (ready_complete)
        for (size_t occurrence = 0; occurrence < cube_sets.size(); ++occurrence)
          for (int lane = 0; lane < num_aiv_; ++lane)
            AddEdge(cube_sets[occurrence], lane_waits[lane][occurrence]);
    }
  }

  void AddPipeBarrierEdges() {
    for (int barrier = 0; barrier < N; ++barrier) {
      const auto &event = instances[barrier];
      if (event.kind != EventKind::kPipeBarrier || !UsableSync(event))
        continue;
      for (int index = 0; index < N; ++index) {
        const auto &access = instances[index];
        if (access.Owner() != event.Owner() ||
            (event.exec_pipe != ResourcePipe::kAll &&
             access.exec_pipe != event.exec_pipe))
          continue;
        if (access.issue_order < event.issue_order)
          AddEdge(index, barrier);
        else if (access.issue_order > event.issue_order)
          AddEdge(barrier, index);
      }
    }
  }

  void ComputeTransitiveClosure() {
    const int words = (N + 63) / 64;
    std::vector<uint64_t> bits(static_cast<size_t>(N) * words, 0);
    auto row = [&](int index) {
      return bits.data() + static_cast<size_t>(index) * words;
    };
    for (int from = 0; from < N; ++from)
      for (int to = 0; to < N; ++to)
        if (reachable[from][to])
          row(from)[to >> 6] |= uint64_t(1) << (to & 63);
    for (int via = 0; via < N; ++via)
      for (int from = 0; from < N; ++from)
        if (row(from)[via >> 6] & (uint64_t(1) << (via & 63)))
          for (int word = 0; word < words; ++word)
            row(from)[word] |= row(via)[word];
    for (int from = 0; from < N; ++from)
      for (int to = 0; to < N; ++to)
        reachable[from][to] = (row(from)[to >> 6] >> (to & 63)) & 1ULL;
  }
};

// Pairwise buffer lifetime ordering.

class BufferReuseAnalysis {
public:
  int num_buffers = 0;
  std::vector<const VarNode *> buffer_vars; // index → VarNode*
  bool CanReuse(int lhs, int rhs) const { return compatible[lhs][rhs]; }
  std::vector<std::vector<bool>> compatible;

  void Build(const HappensBeforeGraph &hb,
             const std::unordered_map<const VarNode *, const AllocBufferNode *>
                 &shmem_allocs,
             const std::vector<const VarNode *> &alloc_order) {
    for (const VarNode *v : alloc_order) {
      if (shmem_allocs.count(v))
        buffer_vars.push_back(v);
    }
    if (buffer_vars.size() < shmem_allocs.size()) {
      std::unordered_set<const VarNode *> seen(buffer_vars.begin(),
                                               buffer_vars.end());
      std::vector<const VarNode *> tail;
      for (const auto &kv : shmem_allocs)
        if (!seen.count(kv.first))
          tail.push_back(kv.first);
      std::sort(tail.begin(), tail.end(),
                [](const VarNode *a, const VarNode *b) {
                  if (a->name_hint != b->name_hint)
                    return a->name_hint < b->name_hint;
                  return a < b;
                });
      buffer_vars.insert(buffer_vars.end(), tail.begin(), tail.end());
    }
    num_buffers = static_cast<int>(buffer_vars.size());
    // Collect ALL access nodes per buffer (not first/last per iter): edge
    // formation must consider every access point, otherwise per-iter
    // aggregation can drop accesses that prove lifetime overlap.
    std::vector<std::vector<int>> all_nodes(num_buffers);
    for (int i = 0; i < num_buffers; ++i)
      hb.GetAllBufferAccessNodes(buffer_vars[i], all_nodes[i]);

    compatible.assign(num_buffers, std::vector<bool>(num_buffers, false));
    if (!hb.within_budget || !hb.complete_owner_domain)
      return;
    auto separated = [&](const std::vector<int> &left,
                         const std::vector<int> &right) {
      if (left.empty() || right.empty())
        return true;
      for (int a : left)
        for (int b : right)
          for (int loop : hb.instances[a].loop_ids)
            if (std::find(hb.instances[b].loop_ids.begin(),
                          hb.instances[b].loop_ids.end(),
                          loop) != hb.instances[b].loop_ids.end())
              return false;
      bool forward = true, reverse = true;
      for (int a : left)
        for (int b : right) {
          forward &= hb.reachable[a][b];
          reverse &= hb.reachable[b][a];
        }
      return forward != reverse;
    };
    for (int a = 0; a < num_buffers; ++a) {
      for (int b = a + 1; b < num_buffers; ++b) {
        if (all_nodes[a].empty() || all_nodes[b].empty())
          continue;
        bool private_ub = IsDynamicSharedMemory(GetRef<Var>(buffer_vars[a])) &&
                          IsDynamicSharedMemory(GetRef<Var>(buffer_vars[b]));
        for (int index : all_nodes[a])
          private_ub &= hb.instances[index].core_mask != kCoreUnassigned;
        for (int index : all_nodes[b])
          private_ub &= hb.instances[index].core_mask != kCoreUnassigned;
        bool safe = true;
        if (private_ub) {
          std::map<ExecutionOwner, std::vector<int>> left, right;
          for (int index : all_nodes[a])
            left[hb.instances[index].Owner()].push_back(index);
          for (int index : all_nodes[b])
            right[hb.instances[index].Owner()].push_back(index);
          for (const auto &[owner, accesses] : left)
            safe &= separated(accesses, right[owner]);
        } else {
          safe = separated(all_nodes[a], all_nodes[b]);
        }
        compatible[a][b] = compatible[b][a] = safe;
      }
    }
  }
};

/*! \brief Infer pairwise reuse compatibility for one allocation scope. */
BufferAliasMap
InferBufferAliasesForScope(const Stmt &stmt, const AllocationMap &allocations,
                           const AllocationOrder &allocation_order,
                           const std::string &target_scope, bool verbose,
                           bool disable_reuse,
                           std::optional<AscendArchitecture> architecture) {
  if (disable_reuse || allocations.size() < 2) {
    BufferAliasMap aliases;
    for (const auto &[var, _] : allocations)
      AddBufferAliasStorage(&aliases, GetRef<Var>(var));
    return aliases;
  }
  SyncAwareLinearizer linearizer(target_scope);
  linearizer(stmt);
  if (verbose && !linearizer.unknown_instr_ops.empty()) {
    std::vector<std::string> ops(linearizer.unknown_instr_ops.begin(),
                                 linearizer.unknown_instr_ops.end());
    std::sort(ops.begin(), ops.end());

    LOG(WARNING)
        << "InferBufferAliases: instructions with unknown executing pipe for "
           "scope \""
        << target_scope
        << "\" — their accesses are conservatively excluded from "
           "program-order wiring (some reuse may be missed). Add them to "
           "GetAscendCallPipeMask in ascend_pipe.h:";
    for (const std::string &name : ops)
      LOG(WARNING) << " - " << name;
  }
  const std::vector<LinearEvent> &events = linearizer.linear_events;
  HappensBeforeGraph happens_before;
  happens_before.Build(events, architecture);

  BufferReuseAnalysis reuse_analysis;
  reuse_analysis.Build(happens_before, allocations, allocation_order);

  BufferAliasMap aliases;
  for (const VarNode *var : reuse_analysis.buffer_vars)
    AddBufferAliasStorage(&aliases, GetRef<Var>(var));
  for (int lhs = 0; lhs < reuse_analysis.num_buffers; ++lhs) {
    for (int rhs = lhs + 1; rhs < reuse_analysis.num_buffers; ++rhs) {
      if (reuse_analysis.CanReuse(lhs, rhs)) {
        AddBufferAlias(&aliases, GetRef<Var>(reuse_analysis.buffer_vars[lhs]),
                       GetRef<Var>(reuse_analysis.buffer_vars[rhs]));
      }
    }
  }
  ValidateBufferAliasMap(aliases);

  if (verbose) {
    LOG(DEBUG) << "Manual-schedule buffer-alias contract for scope \""
               << target_scope << "\":";
    for (int a = 0; a < reuse_analysis.num_buffers; ++a) {
      std::stringstream stream;
      stream << "    " << reuse_analysis.buffer_vars[a]->name_hint << " -> {";
      for (int b = 0; b < reuse_analysis.num_buffers; ++b) {
        if (a != b && reuse_analysis.CanReuse(a, b)) {
          stream << reuse_analysis.buffer_vars[b]->name_hint << " ";
        }
      }
      stream << "}";
      LOG(DEBUG) << stream.str();
    }
  }
  return aliases;
}

void MergeBufferAliasMaps(BufferAliasMap *destination,
                          const BufferAliasMap &source) {
  for (const auto &[storage, _] : source) {
    ICHECK(!destination->count(storage))
        << "Buffer storage appears in more than one allocation scope: "
        << storage;
    AddBufferAliasStorage(destination, storage);
  }
  for (const auto &[storage, compatible] : source) {
    for (const Var &other : compatible) {
      if (!HasBufferAlias(*destination, storage, other))
        AddBufferAlias(destination, storage, other);
    }
  }
}

class BufferAliasContractStripper : public StmtMutator {
private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kBufferAliasMap)
      return VisitStmt(op->body);
    return StmtMutator::VisitStmt_(op);
  }
};

Stmt InferBufferAliases(Stmt stmt, bool verbose, bool disable_reuse,
                        std::optional<AscendArchitecture> architecture) {
  stmt = BufferAliasContractStripper()(std::move(stmt));
  AllocateCollector collector;
  collector(stmt);
  BufferAliasMap aliases;
  auto analyze_scope = [&](const AllocationMap &allocations,
                           const AllocationOrder &allocation_order,
                           const std::string &target_scope) {
    if (allocations.empty())
      return;
    BufferAliasMap local = InferBufferAliasesForScope(
        stmt, allocations, allocation_order, target_scope, verbose,
        disable_reuse, architecture);
    MergeBufferAliasMaps(&aliases, local);
  };

  analyze_scope(collector.dyn_shmem_allocs, collector.dyn_shmem_order, "");
  std::vector<std::string> scopes;
  scopes.reserve(collector.ascend_scope_allocs.size());
  for (const auto &[scope, _] : collector.ascend_scope_allocs)
    scopes.push_back(scope);
  std::sort(scopes.begin(), scopes.end());
  for (const std::string &scope : scopes) {
    analyze_scope(collector.ascend_scope_allocs.at(scope),
                  collector.ascend_scope_order.at(scope), scope);
  }
  ValidateBufferAliasMap(aliases);
  return AttrStmt(std::move(aliases), kBufferAliasMap, Integer(1),
                  std::move(stmt));
}

} // namespace

using namespace tirx::transform;

namespace transform {

Pass InferBufferAliases() {
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext context) {
    bool verbose =
        context
            ->GetConfig<Bool>(kDebugMergeSharedMemoryAllocations, Bool(false))
            .value();
    bool disable_reuse =
        context->GetConfig<Bool>(kDisableSharedMemoryReuse, Bool(false))
            .value();
    Target target = func->GetAttr<Target>(tvm::attr::kTarget)
                        .value_or(Target::Current(true));
    std::optional<AscendArchitecture> architecture;
    if (target.defined())
      architecture = GetAscendTargetInfo(target).architecture;
    PrimFuncNode *writer = func.CopyOnWrite();
    writer->body = tl::InferBufferAliases(std::move(writer->body), verbose,
                                          disable_reuse, architecture);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferBufferAliases", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.InferBufferAliases", InferBufferAliases);
}

} // namespace transform
} // namespace tl
} // namespace tvm
