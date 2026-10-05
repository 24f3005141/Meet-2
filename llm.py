import json
import os
import re
from google import genai
from google.genai import types

_client = None

# Auto-updating alias -- always points at Google's current best Flash
# model, so this never breaks when a dated model (e.g. gemini-2.5-flash,
# which is being shut down 16 October 2026) gets retired. Pin to a dated
# model ONLY if you need output to stay stable across model upgrades.
GEMINI_MODEL = "gemini-flash-latest"


def get_client():
    """
    Lazy singleton -- only creates the Gemini client the first time it's
    actually needed. Reads GEMINI_API_KEY from the environment
    automatically (loaded from .env by app.py's load_dotenv() call).
    """
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


# P3's exact engineered prompt (v3.5), per docs/GENERATE_VALIDATE.md step 1:
# loaded from prompts/roadmap_prompt_v3_7.txt rather than inlined here, so
# a prompt revision is a file swap, not a code change. DO NOT paraphrase or
# "clean up" the file's wording when updating it -- every rule was written
# to suppress a specific failure mode the model kept producing. This is
# model-agnostic plain text, so it needs no changes for the Gemini switch.
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
    on the backticks. Gemini does this at least as often as Claude did.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_llm_for_roadmap(profile_dict, retry_instruction=None):
    """
    Makes one call to the Gemini API and returns the raw text response.

    retry_instruction: if the previous attempt failed (bad JSON or failed
    validation), pass a description of what went wrong here -- it gets
    appended to the user message so the model can self-correct on the
    next attempt.
    """
    client = get_client()

    # The system prompt (P3's) ends with "STUDENT_PROFILE:" expecting the
    # raw JSON right after it -- so the user message (Gemini's "contents")
    # is JUST the profile, no extra wrapper text, to match how P3
    # engineered/tested it against Claude originally.
    user_message = json.dumps(profile_dict, indent=2)
    if retry_instruction:
        user_message += "\n\nIMPORTANT -- fix this before responding: " + retry_instruction

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=ROADMAP_SYSTEM_PROMPT,
            # Bumped from a smaller default -- the output contract
            # (milestones with skills[]/projects[] per phase,
            # internship_strategy, tradeoffs) is sizeable.
            max_output_tokens=4000,
        ),
    )

    # Gemini's response.text concatenates all text parts for you --
    # simpler than Claude's response.content list of typed blocks.
    return response.text


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


def _history_to_gemini_contents(conversation_history, user_message):
    """
    Converts our {"role": "user"|"assistant", "content": "..."} history
    format into Gemini's types.Content list. Gemini uses role "model"
    where Claude/OpenAI use "assistant" -- that's the one real format
    difference callers need not worry about.
    """
    contents = []
    for turn in (conversation_history or []):
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    return contents


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

    contents = _history_to_gemini_contents(conversation_history, user_message)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=500,
        ),
    )

    return response.text


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