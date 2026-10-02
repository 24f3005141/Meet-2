"""
tools/validate_roadmap.py

Per docs/GENERATE_VALIDATE.md step 3: the backend must not trust one raw
LLM output. This validates a generated roadmap against:

  1. schema/roadmap_output.schema.json -- structural correctness
     (Dhruv's exact schema file, additionalProperties: false everywhere).
  2. A set of business-rule checks mirroring prompts/roadmap_prompt_v3_7.txt's
     hard-fail checklist -- things a JSON Schema literally cannot express
     (banned phrases, path diversity, target-role relevance, destination
     matching the profile's stated career).

These business-rule checks are PATTERN-BASED HEURISTICS, not a full
semantic re-implementation of the prompt's rules -- they catch the most
common, mechanically-detectable failure modes (Tests 010-012 in the prompt
doc), not every possible violation. Treat a "VALID" result as "passed our
automated checks", not as "a human has reviewed this roadmap".

Usage as a module:
    from tools.validate_roadmap import validate_roadmap
    ok, failures = validate_roadmap(roadmap_dict, profile_dict)

Usage as a CLI:
    python tools/validate_roadmap.py roadmap.json profile.json
"""
import json
import os
import sys

from jsonschema import validate as jsonschema_validate, ValidationError

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "schema", "roadmap_output.schema.json"
)

# RULE 17B (banned skill/capability words) + RULE 21B (banned cost wording)
BANNED_PHRASES = [
    "master", "mastery", "mastering",
    "advanced level", "advanced python", "advanced ml",
    "job-ready", "expert-level", "large-scale",
    "manageable costs", "merit-based", "keeping costs down",
]

# RULE 6C / 6D -- target_roles must stay adjacent to ML/AI/data/software
BANNED_TARGET_ROLE_KEYWORDS = [
    "quant", "trading", "finance", "support",
    "sales", "marketing", "medical", "nurse", "doctor",
]

# RULE 18B -- projects must read as future goals, not completed work
PAST_TENSE_PROJECT_STARTS = [
    "completed", "contributed", "refactored", "finished", "did ",
]


def _collect_strings(obj):
    """Walks a JSON-like structure and yields every string value in it."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _collect_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _collect_strings(v)


def _check_schema(roadmap):
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    try:
        jsonschema_validate(instance=roadmap, schema=schema)
        return []
    except ValidationError as e:
        location = "/".join(str(p) for p in e.path) or "(root)"
        return [f"SCHEMA: at '{location}': {e.message}"]


def _check_banned_phrases(roadmap):
    failures = []
    for s in _collect_strings(roadmap):
        low = s.lower()
        for phrase in BANNED_PHRASES:
            if phrase in low:
                failures.append(f"RULE17B/21B: banned phrase '{phrase}' found in: \"{s}\"")
    return failures


def _check_no_internship_word_in_projects(roadmap):
    failures = []
    for path in roadmap.get("paths", []):
        for milestone in path.get("milestones", []):
            for project in milestone.get("projects", []):
                if "internship" in project.lower():
                    failures.append(
                        f"RULE18C: 'internship' found inside a projects item: \"{project}\""
                    )
    return failures


def _check_future_tense_projects(roadmap):
    failures = []
    for path in roadmap.get("paths", []):
        for milestone in path.get("milestones", []):
            for project in milestone.get("projects", []):
                low = project.strip().lower()
                if any(low.startswith(bad) for bad in PAST_TENSE_PROJECT_STARTS):
                    failures.append(
                        f"RULE18B: project item looks past-tense, not a future goal: \"{project}\""
                    )
    return failures


def _check_target_roles(roadmap):
    failures = []
    for path in roadmap.get("paths", []):
        roles = path.get("internship_strategy", {}).get("target_roles", [])
        for role in roles:
            low = role.lower()
            for bad in BANNED_TARGET_ROLE_KEYWORDS:
                if bad in low:
                    failures.append(
                        f"RULE6C/6D: target_role '{role}' looks unrelated to the ML/AI/data/software destination"
                    )
    return failures


def _check_path_diversity(roadmap):
    """
    RULE 2C: paths must not share the same branch family. This checks
    exact-string duplication (same degree or same branch text across
    paths) -- it will NOT catch subtler overlap like "AI/ML" vs
    "Artificial Intelligence" being effectively the same family. That
    requires human/LLM judgment, not a string match.
    """
    failures = []
    paths = roadmap.get("paths", [])
    branches = [p.get("education", {}).get("branch", "").strip().lower() for p in paths]
    degrees = [p.get("education", {}).get("degree", "").strip().lower() for p in paths]
    if len(branches) != len(set(branches)):
        failures.append("RULE2C: two or more paths share the exact same branch")
    if len(degrees) != len(set(degrees)):
        failures.append("RULE2C: two or more paths share the exact same degree")
    return failures


def _check_destination_matches_profile(roadmap, profile):
    failures = []
    stated_career = profile.get("student", {}).get("goal", {}).get("career", "").strip().lower()
    destination = roadmap.get("destination", "").strip().lower()
    if stated_career and destination and stated_career != destination:
        failures.append(
            f"RULE1: destination '{roadmap.get('destination')}' does not match "
            f"the profile's stated career '{profile.get('student', {}).get('goal', {}).get('career')}'"
        )
    return failures


def validate_roadmap(roadmap, profile=None):
    """
    Returns (ok: bool, failures: list[str]).

    If schema validation itself fails, the business-rule checks are
    skipped -- there's no point checking banned phrases on a document
    that isn't even shaped correctly yet.

    profile is optional: pass it to also check RULE 1 (destination must
    match the profile's stated career). Omit it (e.g. for the manual
    /api/roadmap testing endpoint, which isn't tied to a real profile)
    to skip that one check.
    """
    failures = _check_schema(roadmap)
    if failures:
        return False, failures

    failures += _check_banned_phrases(roadmap)
    failures += _check_no_internship_word_in_projects(roadmap)
    failures += _check_future_tense_projects(roadmap)
    failures += _check_target_roles(roadmap)
    failures += _check_path_diversity(roadmap)
    if profile is not None:
        failures += _check_destination_matches_profile(roadmap, profile)

    return (len(failures) == 0), failures


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/validate_roadmap.py roadmap.json [profile.json]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        roadmap_data = json.load(f)

    profile_data = None
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            profile_data = json.load(f)

    is_valid, failure_list = validate_roadmap(roadmap_data, profile_data)
    if is_valid:
        print("VALID")
        sys.exit(0)
    else:
        print("INVALID")
        for item in failure_list:
            print(" -", item)
        sys.exit(1)
