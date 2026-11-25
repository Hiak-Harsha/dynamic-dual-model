"""
tests/test_v5_procedural_skills.py — Tests for V5 Procedural Memory, Input Diversity Ladder, and Skill Versioning.

Per V5 Spec Sections 14, 15, 16, 17, 18:
- A Skill is NOT a stored answer; it is a reusable execution procedure.
- Promotion requires both verified successes AND input diversity.
- Consecutive failure clusters trigger demotion.
- Revising a skill creates v2 preserving failure evidence and historical versions.
"""

from __future__ import annotations
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from procedural_memory import ProceduralMemory, Skill


def test_procedural_skill_schema_and_preconditions():
    """Validates that Skill contains procedure steps, preconditions, and recovery strategies."""
    skill = Skill(
        id="csv_to_json_converter",
        name="CSV to JSON Converter",
        trigger_patterns=[r"\bconvert\b.*\bcsv\s+to\s+json\b"],
        semantic_intent="Parse comma-separated data into structured JSON objects",
        preconditions=["Input contains header row and comma-separated rows"],
        steps=["Read CSV input", "Split lines by comma", "Map rows to header keys", "Serialize to JSON"],
        procedure="deterministic_csv_conversion",
        verification="structural",
        required_tools=[],
        version="1.0",
    )

    assert skill.id == "csv_to_json_converter"
    assert len(skill.steps) == 4
    assert "Input contains header row" in skill.preconditions[0]
    assert skill.version == "1.0"
    assert skill.status == "candidate"


def test_input_diversity_promotion_ladder():
    """Validates that promotion to 'eligible' requires diverse input variants, not just repeat identical queries."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_storage = f.name

    try:
        mem = ProceduralMemory(
            storage_path=tmp_storage,
            min_verified_trusted=3,
            min_verified_eligible=5,
            min_variants_eligible=3,
        )

        skill_id = "unit_converter_proc"
        mem.register_candidate(
            skill_id=skill_id,
            trigger_patterns=[r"\bconvert\b.*\b(km|miles|kg|lbs)\b"],
            procedure="unit_conversion_table",
            verification="deterministic",
        )
        assert mem.skills[skill_id].status == "candidate"

        # 3 verified successes with identical input -> promotes to trusted
        for _ in range(3):
            mem.reinforce(skill_id, input_variant="convert 10 km to miles")
        assert mem.skills[skill_id].status == "trusted"
        assert mem.skills[skill_id].distinct_variant_count == 1

        # 2 more verified successes with SAME input (total 5) -> stays trusted because diversity < 3
        mem.reinforce(skill_id, input_variant="convert 10 km to miles")
        mem.reinforce(skill_id, input_variant="convert 10 km to miles")
        assert mem.skills[skill_id].verified_success_count == 5
        assert mem.skills[skill_id].status == "trusted"

        # Introduce distinct input variants
        mem.reinforce(skill_id, input_variant="convert 50 kg to lbs")
        mem.reinforce(skill_id, input_variant="convert 100 miles to km")
        assert mem.skills[skill_id].distinct_variant_count == 3
        # Now promoted to eligible!
        assert mem.skills[skill_id].status == "eligible"
    finally:
        if os.path.exists(tmp_storage):
            os.remove(tmp_storage)


def test_skill_demotion_and_versioning():
    """Validates demotion on consecutive failures and revision to v2 with archive evidence."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_storage = f.name

    try:
        mem = ProceduralMemory(storage_path=tmp_storage)
        skill_id = "markdown_table_formatter"
        mem.register_candidate(
            skill_id=skill_id,
            trigger_patterns=[r"\bformat\b.*\btable\b"],
            procedure="v1: simple split by pipe",
            verification="structural",
        )

        # Promote to trusted
        for i in range(3):
            mem.reinforce(skill_id, input_variant=f"variant {i}")
        assert mem.skills[skill_id].status == "trusted"

        # 2 consecutive failures demotes to candidate
        mem.penalize(skill_id, failure_evidence="Escaped pipes in table cells broke alignment")
        mem.penalize(skill_id, failure_evidence="Escaped pipes in table cells broke alignment")
        assert mem.skills[skill_id].status == "candidate"

        # Revise skill to v2 with improved procedure
        revised = mem.revise_skill(
            skill_id=skill_id,
            failure_evidence="Escaped pipes in table cells caused column misalignment in v1",
            revised_procedure="v2: regex-aware pipe splitter with escape lookbehind",
            revised_steps=["Scan text with regex lookbehind for non-escaped pipes", "Align column widths"],
        )

        assert revised.version == "2.0"
        assert revised.status == "candidate"
        assert revised.consecutive_failures == 0
        assert len(revised.revision_history) == 1
        assert revised.revision_history[0]["version"] == "1.0"
        assert "Escaped pipes" in revised.revision_history[0]["failure_evidence"]
    finally:
        if os.path.exists(tmp_storage):
            os.remove(tmp_storage)
