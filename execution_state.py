"""
execution_state.py — Rich execution tracking, evidence provenance, task frontier, and budget state.

Per V6 Specification Sections V, W, X, Y, AE:
- Evidence Provenance: Every important item carries value, evidence_level (VERIFIED_BY_METHOD),
  source, verification_method, timestamp, artifact_hash, metadata.
- "Verified does not mean absolute truth": Treat verified state as established evidence
  unless contradicted, stale, or insufficiently verified for the current task.
- Task Frontier: completed, current, next, blocked, remaining.
- State Freshness: observed_at, source, content_hash. If underlying artifact changes, invalidate dependent facts.
- Recomputation: Canonical JSON argument sorting json.dumps(..., sort_keys=True, separators=(',', ':')).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import os
import time
import uuid

from base import ExecutionBudget, TaskUsage, count_tokens


class TaskStatus(str):
    """String subclass allowing backwards-compatible equality checks between 'completed' and 'completed_verified'."""
    __hash__ = str.__hash__

    def __eq__(self, o: Any) -> bool:
        if str(self) == str(o):
            return True
        if str(self) in ("completed_verified", "completed") and str(o) in ("completed_verified", "completed"):
            return True
        return False

    def __ne__(self, o: Any) -> bool:
        return not self.__eq__(o)


COMPLETED_VERIFIED = TaskStatus("completed_verified")
COMPLETED_UNVERIFIED = "completed_unverified"
HANDOFF_REQUIRED = "handoff_required"
RECOVERY_ATTEMPTED = "recovery_attempted"
ESCALATION_REQUIRED = "escalation_required"
FAILED_TERMINAL = "failed_terminal"


class EvidenceLevel(str, Enum):
    VERIFIED = "VERIFIED"
    VERIFIED_BY_METHOD = "VERIFIED_BY_METHOD"
    OBSERVED = "OBSERVED"
    ASSUMED = "ASSUMED"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"
    ERROR = "ERROR"


@dataclass
class ArtifactSnapshot:
    """Unified physical disk telemetry snapshot for files touched during execution per V8.1 Section 11."""
    path: str
    sha256: str
    size: int
    artifact_mtime_ns: int
    observed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    operation: str = "read"
    source: str = "tool"
    mtime_ns: Optional[int] = None

    def __post_init__(self):
        if self.mtime_ns is None:
            self.mtime_ns = self.artifact_mtime_ns

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["content_hash"] = self.sha256  # Backward-compatible alias
        d["mtime_ns"] = self.artifact_mtime_ns
        return d


@dataclass
class ExecutionAction:
    """Structured action model for action-level recomputation and graph tracking per V8.1 Section 14."""
    action_id: str
    phase: str = "execution"  # "muscle" | "brain" | "verification" | "continuation"
    action_type: str = "tool_call"
    canonical_tool: str = ""
    canonical_arguments: str = ""
    artifact_dependencies: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StateItem:
    """Individual classified knowledge item with full evidence provenance."""
    value: str
    level: str = EvidenceLevel.OBSERVED.value
    source: str = "execution"
    verification_method: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    artifact_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonical_tool_args(arguments: Any) -> str:
    """Canonicalizes tool arguments into deterministic JSON string."""
    if isinstance(arguments, dict):
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return str(arguments)


@dataclass
class ExecutionState:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    objective: str = ""
    status: str = "running"  # "running" | "completed_verified" | "completed_unverified" | "partial" | "handoff_required" | "blocked" | "failed"
    completed_steps: List[str] = field(default_factory=list)

    # Task Frontier per V6 Section AE
    current_step: Optional[str] = None
    next_action: Optional[str] = None
    blocked_on: Optional[str] = None
    remaining_steps: List[str] = field(default_factory=list)

    # Classified Evidence Collections & Provenance per V6 Section W & X
    verified_facts: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    inferences: List[str] = field(default_factory=list)
    decisions: List[Dict[str, str]] = field(default_factory=list)  # [{"choice": ..., "rationale": ...}]
    items: List[StateItem] = field(default_factory=list)

    # Tool Execution State & Provenance per V6 Section AC & AD
    tool_actions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[ExecutionAction] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    action_signatures: List[str] = field(default_factory=list)
    tool_state: Dict[str, Any] = field(default_factory=dict)
    tool_failures: Dict[str, int] = field(default_factory=dict)

    # State Freshness Tracking per V6 Section Y
    state_freshness: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Resolution & Defect Tracking
    partial_result: str = ""
    unresolved: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    verification_records: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Procedure Information
    procedure_id: Optional[str] = None
    procedure_version: Optional[str] = None
    confidence: float = 1.0

    # Resource Bounds & Usage per V6 Section AA & F
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    usage: TaskUsage = field(default_factory=TaskUsage)

    # Metrics & Timing per V8.1 Section 2
    reuse_count: int = 0
    recomputation_count: int = 0
    recomputation_types: Dict[str, int] = field(default_factory=lambda: {
        "new": 0,
        "reuse": 0,
        "recomputation": 0,
        "verification_repeat": 0,
        "legitimate_repeat": 0,
    })
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    start_time_perf: float = field(default_factory=time.perf_counter)
    elapsed_wall_time: float = 0.0

    def state_token_count(self) -> int:
        """Estimates total tokens currently consumed by this execution state."""
        block = self.to_handoff_block(max_tokens=4000)
        return count_tokens(block)

    def add_verified_fact(
        self,
        fact: str,
        source: str = "verification",
        method: str = "deterministic",
        artifact_hash: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Adds a verified fact with full provenance.
        Per V6 Section X: 'VERIFIED DOES NOT MEAN ABSOLUTE TRUTH'.
        """
        if fact and fact not in self.verified_facts:
            self.verified_facts.append(fact)
            self.items.append(
                StateItem(
                    value=fact,
                    level=EvidenceLevel.VERIFIED.value,
                    source=source,
                    verification_method=method,
                    artifact_hash=artifact_hash,
                    metadata=meta or {},
                )
            )

    def add_observation(
        self,
        obs: str,
        source: str = "execution",
        artifact_hash: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if obs and obs not in self.observations:
            self.observations.append(obs)
            self.items.append(
                StateItem(
                    value=obs,
                    level=EvidenceLevel.OBSERVED.value,
                    source=source,
                    artifact_hash=artifact_hash,
                    metadata=meta or {},
                )
            )

    def add_assumption(self, assumption: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if assumption and assumption not in self.assumptions:
            self.assumptions.append(assumption)
            self.items.append(
                StateItem(
                    value=assumption,
                    level=EvidenceLevel.ASSUMED.value,
                    source="hypothesis",
                    metadata=meta or {},
                )
            )

    def add_inference(self, inference: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if inference and inference not in self.inferences:
            self.inferences.append(inference)
            self.items.append(
                StateItem(
                    value=inference,
                    level=EvidenceLevel.INFERRED.value,
                    source="reasoning",
                    metadata=meta or {},
                )
            )

    def add_decision(self, choice: str, why: str = "", rationale: str = "") -> None:
        reason = why or rationale
        self.decisions.append({"choice": choice, "why": reason, "rationale": reason})

    def add_completed_step(self, step: str) -> None:
        if step and step not in self.completed_steps:
            self.completed_steps.append(step)

    def set_frontier(
        self,
        current: Optional[str] = None,
        next_action: Optional[str] = None,
        blocked: Optional[str] = None,
        remaining: Optional[List[str]] = None,
    ) -> None:
        """Updates the active task execution frontier."""
        if current is not None:
            self.current_step = current
        if next_action is not None:
            self.next_action = next_action
        if blocked is not None:
            self.blocked_on = blocked
        if remaining is not None:
            self.remaining_steps = list(remaining)

    def mark_unresolved(self, item: str) -> None:
        if item and item not in self.unresolved:
            self.unresolved.append(item)

    def add_error(self, err: str) -> None:
        if err and err not in self.errors:
            self.errors.append(err)
            self.items.append(
                StateItem(
                    value=err,
                    level=EvidenceLevel.ERROR.value,
                    source="error",
                )
            )

    def record_state_freshness(self, path: str, content: Any, source: str = "tool", operation: str = "read") -> ArtifactSnapshot:
        """Records content hash and physical disk metadata for artifact freshness tracking via ArtifactSnapshot per V8.1 Section 11."""
        if isinstance(content, ArtifactSnapshot):
            snapshot = content
        else:
            text_str = str(content)
            c_hash = hashlib.sha256(text_str.encode("utf-8")).hexdigest()
            size = len(text_str.encode("utf-8"))
            artifact_mtime_ns = time.time_ns()
            try:
                from tools import _resolve_in_sandbox
                resolved = _resolve_in_sandbox(path)
                if resolved and os.path.exists(resolved):
                    st = os.stat(resolved)
                    artifact_mtime_ns = st.st_mtime_ns
                    size = st.st_size
            except Exception:
                pass

            snapshot = ArtifactSnapshot(
                path=path,
                sha256=c_hash,
                size=size,
                artifact_mtime_ns=artifact_mtime_ns,
                observed_at=datetime.now(timezone.utc).isoformat(),
                operation=operation,
                source=source,
            )
        self.state_freshness[path] = snapshot.to_dict()
        return snapshot

    def is_artifact_fresh(self, path: str, current_content: str) -> bool:
        """Checks whether the cached observation for path matches current content."""
        if path not in self.state_freshness:
            return False
        curr_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
        entry = self.state_freshness[path]
        return entry.get("sha256") == curr_hash or entry.get("content_hash") == curr_hash

    def invalidate_artifact(self, path: str) -> None:
        """Invalidates facts and observations dependent on an updated artifact."""
        if path in self.state_freshness:
            del self.state_freshness[path]
        # Invalidate dependent facts
        self.verified_facts = [f for f in self.verified_facts if path not in f]
        self.items = [i for i in self.items if not (i.artifact_hash and path in i.value)]

    def record_tool_action(
        self,
        action: Dict[str, Any],
        phase: str = "execution",
        action_type: str = "tool_call",
        dependencies: Optional[List[str]] = None,
    ) -> ExecutionAction:
        self.tool_actions.append(action)
        self.usage.tool_calls += 1
        name = action.get("name", "unknown") if isinstance(action, dict) else str(action)
        args = action.get("arguments", {}) if isinstance(action, dict) else {}
        canonical_args = canonical_tool_args(args)
        sig = f"{name}({canonical_args})"
        self.action_signatures.append(sig)

        dep_list = list(dependencies or [])
        if isinstance(args, dict) and "path" in args and args["path"] not in dep_list:
            dep_list.append(str(args["path"]))

        act = ExecutionAction(
            action_id=action.get("id") or f"act_{len(self.actions)+1}",
            phase=phase,
            action_type=action_type,
            canonical_tool=name,
            canonical_arguments=canonical_args,
            artifact_dependencies=dep_list,
        )
        self.actions.append(act)
        return act

    def calculate_action_recomputation(self, baseline_actions: List[str]) -> Tuple[int, float]:
        """Calculates duplicate action count and repetition ratio against a baseline action sequence."""
        if not baseline_actions:
            return 0, 0.0
        repeats = sum(1 for a in self.action_signatures if a in baseline_actions)
        ratio = repeats / max(1, len(self.action_signatures))
        return repeats, ratio

    def calculate_action_recomputation_metrics(
        self, baseline_actions: Any
    ) -> Dict[str, Any]:
        """Computes rich action-level recomputation metrics per V8.1 Section 14."""
        base_sigs: List[str] = []
        if isinstance(baseline_actions, list):
            for item in baseline_actions:
                if isinstance(item, ExecutionAction):
                    base_sigs.append(f"{item.canonical_tool}({item.canonical_arguments})")
                elif isinstance(item, dict):
                    name = item.get("canonical_tool") or item.get("name", "")
                    args = item.get("canonical_arguments") or canonical_tool_args(item.get("arguments", {}))
                    base_sigs.append(f"{name}({args})")
                else:
                    base_sigs.append(str(item))

        current_sigs = [f"{a.canonical_tool}({a.canonical_arguments})" for a in self.actions] or list(self.action_signatures)

        unique_useful = len(set(current_sigs))
        reused = self.reuse_count
        recomputed = 0
        verif_repeats = 0
        legit_repeats = 0

        # Check against baseline signatures with sequential accounting
        matched_base = list(base_sigs)
        for act in self.actions:
            sig = f"{act.canonical_tool}({act.canonical_arguments})"
            if act.phase == "verification":
                verif_repeats += 1
            elif sig in matched_base:
                matched_base.remove(sig)
                recomputed += 1

        return {
            "unique_useful_actions": unique_useful,
            "reused_actions": reused,
            "recomputed_actions": recomputed,
            "verification_repeats": verif_repeats,
            "legitimate_repeats": legit_repeats,
            "total_actions": len(current_sigs),
            "recomputation_ratio": recomputed / max(1, len(current_sigs)),
        }

    @property
    def frontier(self) -> Dict[str, Any]:
        """Exposes Task Frontier as a dictionary per V6 Section AE and V8.1 Section 13."""
        artifacts = list(self.state_freshness.keys())
        all_deps = []
        for act in self.actions:
            for d in act.artifact_dependencies:
                if d not in all_deps:
                    all_deps.append(d)

        return {
            "completed_steps": list(self.completed_steps),
            "current_step": self.current_step,
            "current": self.current_step,
            "next_action": self.next_action,
            "remaining_steps": list(self.remaining_steps),
            "remaining": list(self.remaining_steps),
            "blocked_on": self.blocked_on,
            "blocked": self.blocked_on,
            "dependencies": all_deps,
            "observations": list(self.observations),
            "artifacts": artifacts,
        }

    def record_tool_result(self, result: Dict[str, Any]) -> None:
        self.tool_results.append(result)

    def get_canonical_tool_signature(self, tool_name: str, arguments: Any) -> str:
        """Returns deterministic signature for tool invocation."""
        return f"{tool_name}:{canonical_tool_args(arguments)}"

    def check_tool_reuse(self, tool_name: str, arguments: Any) -> Tuple[bool, Optional[str], str]:
        """Per V8.1 Section 2:
        Distinguish NEW vs REUSE vs RECOMPUTATION vs VERIFICATION_REPEAT vs LEGITIMATE_REPEAT.
        Checks physical disk state and hashes before permitting reuse.
        Returns (can_reuse, previous_output, repetition_category).
        """
        sig = self.get_canonical_tool_signature(tool_name, arguments)
        canonical_args = canonical_tool_args(arguments)

        matching_results = []
        for r in self.tool_results:
            r_name = r.get("name") or r.get("tool", "")
            if r_name != tool_name:
                continue
            r_sig = r.get("tool_signature")
            r_canonical = r.get("canonical_arguments")
            r_args = r.get("arguments")

            if r_sig == sig:
                matching_results.append(r)
            elif r_canonical is not None and r_canonical == canonical_args:
                matching_results.append(r)
            elif r_args is not None and self.get_canonical_tool_signature(r_name, r_args) == sig:
                matching_results.append(r)
            elif r_args is None and r_canonical is None:
                matching_results.append(r)

        if not matching_results:
            return False, None, "new"

        last_result = matching_results[-1]
        out = last_result.get("output") or last_result.get("result", "")

        # Strict check: check structured error indicators only per V8.1 Section 9
        if last_result.get("success") is False or last_result.get("error") or last_result.get("error_code"):
            return False, None, "legitimate_repeat"

        # If read_file, physically verify the actual artifact on disk!
        if tool_name == "read_file" and isinstance(arguments, dict) and "path" in arguments:
            path = arguments["path"]
            from tools import _resolve_in_sandbox, get_file_content_hash
            resolved = _resolve_in_sandbox(path)
            if resolved is None or not os.path.exists(resolved):
                return False, None, "legitimate_repeat"

            recorded = self.state_freshness.get(path) or self.state_freshness.get(resolved)
            if not recorded:
                return False, None, "legitimate_repeat"

            current_hash = get_file_content_hash(path)
            recorded_hash = recorded.get("sha256") or recorded.get("content_hash")
            if current_hash != recorded_hash:
                # Disk content has changed: invalidates prior observation
                return False, None, "legitimate_repeat"

            # Disk hash matches recorded observation: safe to reuse!
            return True, out, "reuse"

        return True, out, "reuse"

    def is_budget_exceeded(self) -> Tuple[bool, str]:
        """Checks whether any resource dimension has exceeded the ExecutionBudget."""
        # Wall-time limit check
        if self.budget.max_wall_time > 0:
            current_elapsed = self.elapsed_wall_time + (time.perf_counter() - self.start_time_perf)
            if current_elapsed > self.budget.max_wall_time:
                return True, f"Wall-time limit exceeded ({current_elapsed:.2f}s/{self.budget.max_wall_time}s)"

        # State token size check
        if self.budget.max_state_tokens > 0:
            current_tokens = self.state_token_count()
            if current_tokens > self.budget.max_state_tokens:
                return True, f"State token size limit exceeded ({current_tokens}/{self.budget.max_state_tokens})"

        # Model call & token usage limits
        if self.budget.max_model_calls > 0 and self.usage.model_calls >= self.budget.max_model_calls:
            return True, f"Model call limit exceeded ({self.usage.model_calls}/{self.budget.max_model_calls})"
        if self.budget.max_input_tokens > 0 and self.usage.total_input_tokens >= self.budget.max_input_tokens:
            return True, f"Input token limit exceeded ({self.usage.total_input_tokens}/{self.budget.max_input_tokens})"
        if self.budget.max_output_tokens > 0 and self.usage.total_output_tokens >= self.budget.max_output_tokens:
            return True, f"Output token limit exceeded ({self.usage.total_output_tokens}/{self.budget.max_output_tokens})"
        if self.budget.max_tool_calls > 0 and self.usage.tool_calls >= self.budget.max_tool_calls:
            return True, f"Tool call limit exceeded ({self.usage.tool_calls}/{self.budget.max_tool_calls})"
        return False, ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Update elapsed_wall_time before saving
        d["elapsed_wall_time"] = self.elapsed_wall_time + (time.perf_counter() - self.start_time_perf)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionState:
        data_copy = dict(data)
        if "budget" in data_copy and isinstance(data_copy["budget"], dict):
            data_copy["budget"] = ExecutionBudget(**data_copy["budget"])
        if "usage" in data_copy and isinstance(data_copy["usage"], dict):
            data_copy["usage"] = TaskUsage(**data_copy["usage"])
        if "items" in data_copy and isinstance(data_copy["items"], list):
            data_copy["items"] = [StateItem(**i) if isinstance(i, dict) else i for i in data_copy["items"]]
        if "actions" in data_copy and isinstance(data_copy["actions"], list):
            data_copy["actions"] = [ExecutionAction(**a) if isinstance(a, dict) else a for a in data_copy["actions"]]
        obj = cls(**data_copy)
        obj.start_time_perf = time.perf_counter()
        return obj

    def to_handoff_block(self, handoff_reason: str = "", suggested_next_step: str = "", max_tokens: int = 400) -> str:
        """Renders a token-budget-aware structured handoff prompt.

        Prioritization order per V6 Section AF & AG:
        1. objective
        2. handoff reason
        3. verified evidence (treat as established evidence, not unquestionable truth)
        4. execution frontier (completed, current, next, blocked, remaining)
        5. tool history & artifact state
        6. unresolved items / errors
        7. decisions & observations
        8. continuation directive
        """
        lines = [
            "[CONTINUATION HANDOFF FROM MUSCLE MEMORY]",
            f"Objective: {self.objective or '(unspecified)'}",
        ]
        if handoff_reason:
            lines.append(f"Handoff Reason: {handoff_reason}")

        # 1. Verified Facts (Established Evidence)
        if self.verified_facts:
            lines.append("Verified Facts (ESTABLISHED EVIDENCE - Re-evaluate only if contradicted or stale):")
            for vf in self.verified_facts[-6:]:
                lines.append(f"  [VERIFIED] {vf}")

        # 2. Execution Frontier
        frontier_items = []
        if self.current_step:
            frontier_items.append(f"Current Step: {self.current_step}")
        if self.next_action:
            frontier_items.append(f"Next Action: {self.next_action}")
        if self.blocked_on:
            frontier_items.append(f"Blocked On: {self.blocked_on}")
        if self.remaining_steps:
            frontier_items.append(f"Remaining Steps: {', '.join(self.remaining_steps[:4])}")
        if frontier_items:
            lines.append("Execution Frontier:")
            for fi in frontier_items:
                lines.append(f"  > {fi}")

        # 3. Completed steps
        if self.completed_steps:
            lines.append("Completed Steps:")
            for step in self.completed_steps[-6:]:
                lines.append(f"  - {step}")

        # 4. Unresolved items
        if self.unresolved:
            lines.append("Unresolved Items (ACTION REQUIRED):")
            for item in self.unresolved[-4:]:
                lines.append(f"  * {item}")

        # 5. Errors
        if self.errors:
            lines.append("Errors Encountered:")
            for err in self.errors[-3:]:
                lines.append(f"  ! {err}")

        # 6. Artifact & Tool State
        if self.tool_state:
            active_tools = [k for k, v in self.tool_state.items() if v]
            if active_tools:
                lines.append(f"Active Artifact State: {', '.join(active_tools[-6:])}")

        # 7. Decisions
        if self.decisions:
            lines.append("Decisions Made:")
            for dec in self.decisions[-3:]:
                choice = dec.get("choice", "")
                why = dec.get("why") or dec.get("rationale", "")
                lines.append(f"  - {choice}" + (f": {why}" if why else ""))

        # 8. Observations
        unverified_obs = [o for o in self.observations if o not in self.verified_facts]
        if unverified_obs:
            lines.append("Observations (Raw Telemetry):")
            for obs in unverified_obs[-3:]:
                lines.append(f"  - {obs}")

        # 9. Continuation Directive (Footer)
        lines.append(
            "[CONTINUATION DIRECTIVE:\n"
            "1. Continue execution from the Task Frontier.\n"
            "2. Treat verified state as established evidence unless contradicted, stale, or insufficiently verified.\n"
            "3. Do not repeat verified tool operations when fresh results exist.\n"
            "4. Resolve only the remaining work to satisfy the user objective.]"
        )

        rendered = "\n".join(lines)

        # Budget enforcement: compact summary if exceeding max_tokens
        if count_tokens(rendered) > max_tokens:
            compact_lines = [
                "[CONTINUATION HANDOFF FROM MUSCLE MEMORY]",
                f"Objective: {self.objective or '(unspecified)'}",
                f"Handoff Reason: {handoff_reason or 'State transferred'}",
            ]
            if self.verified_facts:
                compact_lines.append(f"Verified Evidence: {'; '.join(self.verified_facts[-3:])}")
            if self.current_step or self.next_action:
                compact_lines.append(f"Frontier: current={self.current_step or 'N/A'}, next={self.next_action or 'N/A'}")
            if self.completed_steps:
                compact_lines.append(f"Completed: {'; '.join(self.completed_steps[-3:])}")
            if self.unresolved:
                compact_lines.append(f"Remaining: {'; '.join(self.unresolved[-3:])}")
            if self.errors:
                compact_lines.append(f"Errors: {'; '.join(self.errors[-2:])}")
            compact_lines.append("[DIRECTIVE: Continue from frontier; reuse verified evidence; complete remaining objective.]")
            rendered = "\n".join(compact_lines)

        return rendered


@dataclass
class HandoffState:
    """Encapsulates the state required for seamless continuation."""
    execution_state: ExecutionState
    handoff_reason: str = "Muscle memory uncertain or encountered complexity"
    evidence: str = ""
    suggested_next_step: str = ""
    procedure_info: Optional[str] = None

    def render_prompt(self, max_tokens: int = 400) -> str:
        return self.execution_state.to_handoff_block(
            handoff_reason=self.handoff_reason,
            suggested_next_step=self.suggested_next_step,
            max_tokens=max_tokens,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_state": self.execution_state.to_dict(),
            "handoff_reason": self.handoff_reason,
            "evidence": self.evidence,
            "suggested_next_step": self.suggested_next_step,
            "procedure_info": self.procedure_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HandoffState:
        return cls(
            execution_state=ExecutionState.from_dict(data["execution_state"]),
            handoff_reason=data.get("handoff_reason", ""),
            evidence=data.get("evidence", ""),
            suggested_next_step=data.get("suggested_next_step", ""),
            procedure_info=data.get("procedure_info"),
        )
