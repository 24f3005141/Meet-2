import json
import os
import re
from anthropic import Anthropic

_client = None


def get_client():
    """
    Lazy singleton -- only creates the Anthropic client the first time
    it's actually needed. Reads ANTHROPIC_API_KEY from the environment
    automatically (loaded from .env by app.py's load_dotenv() call).
    """
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


# P3's exact engineered prompt (v3.5), per docs/GENERATE_VALIDATE.md step 1:
# loaded from prompts/roadmap_prompt_v3_7.txt rather than inlined here, so
# a prompt revision is a file swap, not a code change. DO NOT paraphrase or
# "clean up" the file's wording when updating it -- every rule was written
# to suppress a specific failure mode the model kept producing.
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
ROADMAP_PROMPT_PATH = os.path.join(PROMPT_DIR, "roadmap_prompt_v3_7.txt")


def _load_roadmap_prompt():
    with open(ROADMAP_PROMPT_PATH, "r") as f:
        return f.read()


ROADMAP_SYSTEM_PROMPT = _load_roadmap_prompt()


def strip_code_fences(text):
    """
    LLMs often wrap JSON output in ```json ... ``` even when told not to.
    This strips that wrapping defensively so json.loads() doesn't choke
    on the backticks.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_llm_for_roadmap(profile_dict, retry_instruction=None):
    """
    Makes one call to the Claude API and returns the raw text response.

    retry_instruction: if the previous attempt failed (bad JSON or failed
    schema validation), pass a description of what went wrong here --
    it gets appended to the user message so the model can self-correct
    on the next attempt.
    """
    client = get_client()

    # The system prompt (P3's) ends with "STUDENT_PROFILE:" expecting the
    # raw JSON right after it -- so the user message is JUST the profile,
    # no extra wrapper text, to match how P3 engineered/tested it.
    user_message = json.dumps(profile_dict, indent=2)
    if retry_instruction:
        user_message += "\n\nIMPORTANT -- fix this before responding: " + retry_instruction

    response = client.messages.create(
        model="claude-sonnet-4-6",
        # Bumped from 2000 -- the new output contract (milestones with
        # skills[]/projects[] per phase, internship_strategy, tradeoffs)
        # is significantly larger than the old roadmap_steps list.
        max_tokens=4000,
        system=ROADMAP_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    # response.content is a list of blocks (text, tool_use, etc).
    # We only care about the text blocks, concatenated.
    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return raw_text


COPILOT_SYSTEM_PROMPT_TEMPLATE = """You are Meet 2's career copilot -- a friendly, encouraging assistant \
for an Indian student who just finished 12th grade. You talk like a \
knowledgeable senior/mentor, not a formal advisor. Keep replies short \
(2-5 sentences unless the question genuinely needs more) and concrete.

You have the student's full context below. Use it naturally -- don't \
just repeat it back at them. If their question isn't answerable from \
this context, say so honestly rather than inventing specifics.

STUDENT PROFILE:
{profile_json}

GENERATED ROADMAP:
{roadmap_json}

CURRENT MISSION (what they're working on right now):
{mission_json}
"""


def call_llm_for_copilot(profile_dict, roadmap_dict, mission_dict, user_message, conversation_history=None):
    """
    Conversational call -- unlike call_llm_for_roadmap, this returns
    free-form text, not JSON. conversation_history (optional) is a list
    of {"role": "user"|"assistant", "content": "..."} dicts for
    multi-turn context; pass None or [] for a fresh conversation.
    """
    client = get_client()

    system_prompt = COPILOT_SYSTEM_PROMPT_TEMPLATE.format(
        profile_json=json.dumps(profile_dict, indent=2),
        roadmap_json=json.dumps(roadmap_dict, indent=2) if roadmap_dict else "No roadmap generated yet.",
        mission_json=json.dumps(mission_dict, indent=2) if mission_dict else "No active mission yet.",
    )

    messages = list(conversation_history) if conversation_history else []
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=system_prompt,
        messages=messages,
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return raw_text


def parse_roadmap_json(raw_text):
    """
    Defensively parses the LLM's raw text into a Python dict.
    Raises ValueError with a readable message on failure, so the caller
    can decide whether to retry or give up.
    """
    cleaned = strip_code_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM output as JSON: {e}")