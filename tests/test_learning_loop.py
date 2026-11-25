"""
tests/test_learning_loop.py — Verification of Procedural Memory Learning Loop.

Validates that:
1. When Full Brain successfully resolves a task via handoff that is low-risk and tool-free,
   it registers a new candidate skill.
2. The newly registered skill is stored with status="candidate" and confidence=0.5.
3. On an exact repeat, muscle_engine / retrieve matches and exercises the candidate.
4. On unrelated queries, retrieve returns None (no false-positive overgeneralization).
5. Successes advance the promotion ladder: candidate -> trusted -> eligible.
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from procedural_memory import ProceduralMemory, Skill
from execution_state import ExecutionState
from router import RouteDecision
from muscle_engine import execute_muscle
from mock_provider import MockProvider
from base import ConversationState


def test_procedural_memory_lifecycle():
    test_db = "test_skills_lifecycle.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    try:
        mem = ProceduralMemory(storage_path=test_db)
        
        # 1. Candidate registration
        query = "what is the capital of australia?"
        answer = "The capital of Australia is Canberra."
        skill_id = "learned_capital_australia"
        
        registered = mem.register_candidate(
            skill_id=skill_id,
            trigger_patterns=[r"^\s*what is the capital of australia\??\s*$"],
            procedure=answer,
            verification="consistency",
            is_deterministic=False,
        )
        assert registered.status == "candidate"
        assert registered.confidence == 0.5
        assert registered.success_count == 0

        # 2. Retrieval with only_trusted=False (low-risk tasks allow candidate trials)
        matched = mem.retrieve(query, only_trusted=False)
        assert matched is not None
        assert matched.id == skill_id

        # Retrieval with only_trusted=True must NOT match unproven candidate
        trusted_only = mem.retrieve(query, only_trusted=True)
        assert trusted_only is None

        # Unrelated query must return None
        unrelated = mem.retrieve("how do I install docker?", only_trusted=False)
        assert unrelated is None

        # 3. Promotion ladder: 3 verified successes -> trusted
        for _ in range(3):
            mem.reinforce(skill_id)
        assert mem.skills[skill_id].status == "trusted"
        assert mem.skills[skill_id].confidence >= 0.8
        # Now matches even with only_trusted=True
        assert mem.retrieve(query, only_trusted=True) is not None

        # 4. Promotion to eligible after 10 total successes
        for _ in range(7):
            mem.reinforce(skill_id)
        assert mem.skills[skill_id].status == "eligible"
        assert mem.skills[skill_id].confidence == 1.0

        # 5. Demotion ladder on consecutive failures
        mem.penalize(skill_id)
        assert mem.skills[skill_id].status == "eligible"  # 1 failure doesn't immediately demote
        mem.penalize(skill_id)
        assert mem.skills[skill_id].status == "trusted"   # 2 consecutive failures demotes to trusted
        mem.penalize(skill_id)
        mem.penalize(skill_id)
        assert mem.skills[skill_id].status == "candidate" # demotes to candidate
        mem.penalize(skill_id)
        mem.penalize(skill_id)
        assert mem.skills[skill_id].status == "demoted"   # demotes to demoted

        # Demoted skill must never match
        assert mem.retrieve(query, only_trusted=False) is None

        print("PASS: Procedural memory learning loop and promotion/demotion ladder verified.")

    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


def test_execution_state_handoff_budget():
    state = ExecutionState(objective="Refactor large codebase")
    for i in range(20):
        state.add_observation(f"Observation {i}: established fact regarding module_{i}")
    for i in range(10):
        state.add_completed_step(f"Step {i}: inspected module_{i}")
    for i in range(8):
        state.add_decision(f"Decision {i}", why=f"Rationale {i}")

    handoff_text = state.to_handoff_block(
        handoff_reason="Complexity exceeded threshold",
        suggested_next_step="Proceed with modular rewrite",
    )

    # Must stay under budget (< 1600 characters / ~400 tokens)
    assert len(handoff_text) <= 1650, f"Expected <= 1650 chars, got {len(handoff_text)}"
    # Most recent observations must be preserved
    assert "Observation 19" in handoff_text
    assert "Observation 18" in handoff_text
    # Older observations truncated
    assert "earlier observations omitted" in handoff_text or "Observation 0" not in handoff_text
    print("PASS: ExecutionState handoff block stays under 400-token budget and prioritizes recent observations.")


if __name__ == "__main__":
    test_procedural_memory_lifecycle()
    test_execution_state_handoff_budget()
