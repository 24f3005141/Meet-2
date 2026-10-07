import os
import json
import uuid
import logging
import time
from datetime import date

from flask import Flask, jsonify, request, session
from flask_cors import CORS
from dotenv import load_dotenv

from database import init_db, get_db, get_or_create_user
from validation import validate_against_schema
from llm import (
    call_llm_for_roadmap,
    parse_roadmap_json,
    call_llm_for_copilot,
    get_client,
    GEMINI_MODEL,
)
from tools.validate_roadmap import validate_roadmap as validate_roadmap_full

# Load variables from a .env file (like ANTHROPIC_API_KEY) into os.environ
load_dotenv()

# ---------------------------------------------------------------
# Logging setup
# In production (Render/Railway) stdout is automatically captured and
# shown in the platform's log viewer, so logging.StreamHandler (the
# default) is enough -- no need to write to a file.
# ---------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("meet2")

app = Flask(__name__)
# session (the cookie) needs a secret key to sign itself securely.
# In production this MUST come from an env var, never hardcoded.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
if app.secret_key == "dev-only-change-me":
    logger.warning(
        "FLASK_SECRET_KEY is not set -- using the insecure default. "
        "Set it in .env before deploying."
    )

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions_config.json")

# ---------------------------------------------------------------
# CORS -- P1's frontend almost certainly runs on a different origin
# (different dev server port, or a different deployed domain), so
# without this the browser blocks every request before it even
# reaches Flask. supports_credentials=True lets the cookie-based
# session still work for browser clients that don't explicitly pass
# session_id back (though explicit session_id, as P1's contract uses,
# is the more reliable option across origins).
#
# CORS_ALLOWED_ORIGINS in .env should be a comma-separated list of
# exact frontend URLs in production (e.g. https://meet2.vercel.app).
# "*" (the default here) is fine for hackathon development only.
# ---------------------------------------------------------------
_allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
CORS(
    app,
    origins=_allowed_origins.split(",") if _allowed_origins != "*" else "*",
    supports_credentials=True,
)


# ---------------------------------------------------------------
# Request/response logging
# Logs every request with how long it took and what status came back.
# Useful during the hackathon demo if something breaks on stage --
# you can see exactly which call failed and why, from the platform logs.
# ---------------------------------------------------------------
@app.before_request
def start_timer():
    request._start_time = time.time()


@app.after_request
def log_request(response):
    duration_ms = int((time.time() - getattr(request, "_start_time", time.time())) * 1000)
    logger.info(
        f"{request.method} {request.path} -> {response.status_code} ({duration_ms}ms)"
    )
    return response


# ---------------------------------------------------------------
# Global error handlers
# Without these, an unhandled exception returns Flask's default HTML
# error page -- useless for a JSON API and ugly if a judge's browser
# hits it directly. These make sure every error, expected or not,
# comes back as JSON.
# ---------------------------------------------------------------
@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Not found", "path": request.path}), 404


@app.errorhandler(405)
def handle_405(e):
    return jsonify({"error": f"Method {request.method} not allowed on {request.path}"}), 405


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    # Log the FULL traceback server-side (so you can debug it), but
    # never leak internal details (stack traces, file paths, SQL) to
    # whoever is calling the API.
    logger.exception(f"Unhandled error on {request.method} {request.path}")
    return jsonify({"error": "Something went wrong on our end. Please try again."}), 500


def get_session_user_id():
    """
    Every request needs to know "who is this". We put a random
    session_id in the browser's cookie the first time they show up,
    then look up (or create) their row in the users table.
    """
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return get_or_create_user(session["session_id"])


def resolve_session(client_session_id=None):
    """
    Figures out which session this request belongs to, in order of
    preference:
      1. An explicit session_id the client sent in the request body
         (what /api/submit returns and the client is expected to pass
         back on every later call -- robust across origins, since it
         doesn't depend on cookies surviving a cross-origin request).
      2. The cookie-based session (works fine for same-origin/browser use).
      3. Brand new session_id if neither exists.

    Returns (user_id, session_id) -- always pass session_id back to the
    client in the response so they can store and reuse it.
    """
    if client_session_id:
        sid = client_session_id
    elif "session_id" in session:
        sid = session["session_id"]
    else:
        sid = str(uuid.uuid4())

    session["session_id"] = sid  # keep the cookie in sync too, as a bonus
    user_id = get_or_create_user(sid)
    return user_id, sid


def save_roadmap_to_db(profile_id, roadmap_dict, user_id=None):
    """
    Shared by /api/roadmap (manual save, already-validated JSON) and
    /api/generate-roadmap (LLM-produced, already-validated JSON).
    Assumes the caller already ran validate_against_schema and got None back.
    """
    # Only persist the schema's actual fields (destination + paths) --
    # the caller's dict may also carry a "profile_id" key (the manual
    # /api/roadmap endpoint allows it in the body), which doesn't belong
    # inside the saved roadmap JSON itself.
    clean_roadmap = {
        "destination": roadmap_dict.get("destination"),
        "paths": roadmap_dict["paths"],
    }

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO roadmaps (profile_id, user_id, generated_json) VALUES (?, ?, ?)",
        (profile_id, user_id, json.dumps(clean_roadmap)),
    )
    conn.commit()
    roadmap_id = cur.lastrowid
    conn.close()
    return roadmap_id


def get_latest_roadmap_for_user(user_id):
    """Returns {id, destination, paths, selected_path_index} of this user's most recent roadmap, or None."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, generated_json, selected_path_index FROM roadmaps "
        "WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    parsed = json.loads(row["generated_json"])
    return {
        "id": row["id"],
        "destination": parsed.get("destination"),
        "paths": parsed["paths"],
        "selected_path_index": row["selected_path_index"],  # None until selected
    }


def get_or_create_missions(user_id, roadmap_id, selected_path):
    """
    Idempotent: if missions already exist for this roadmap_id, does
    nothing. Otherwise creates one mission per MILESTONE of the
    STUDENT-SELECTED path (caller is responsible for making sure a
    path has actually been selected before calling this).

    Each milestone is richer than the old roadmap_steps string -- it
    has phase/timeframe/focus/skills/projects -- so the full milestone
    dict is stored in details_json, with "phase" used as the short
    display title.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM missions WHERE roadmap_id = ?", (roadmap_id,))
    already_exist = cur.fetchone()["c"] > 0

    if not already_exist:
        for i, milestone in enumerate(selected_path["milestones"]):
            status = "active" if i == 0 else "pending"
            cur.execute(
                "INSERT INTO missions (user_id, roadmap_id, step_index, title, status, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, roadmap_id, i, milestone["phase"], status, json.dumps(milestone)),
            )
        conn.commit()

    conn.close()


def get_current_mission_row(user_id, roadmap_id):
    """Returns the DB row (sqlite3.Row) for the earliest not-completed mission, or None if all done."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM missions WHERE user_id = ? AND roadmap_id = ? "
        "AND status != 'completed' ORDER BY step_index ASC LIMIT 1",
        (user_id, roadmap_id),
    )
    row = cur.fetchone()
    conn.close()
    return row


def mission_row_to_dict(row):
    result = {
        "mission_id": row["id"],
        "title": row["title"],
        "step_index": row["step_index"],
        "status": row["status"],
    }
    # details_json carries the full milestone (timeframe/focus/skills/
    # projects) -- older rows created before this column existed will
    # have it as None, so fall back gracefully instead of crashing.
    if row["details_json"]:
        result["details"] = json.loads(row["details_json"])
    return result


def log_activity(user_id):
    """Adds/increments today's row in activity_log for this user."""
    today = date.today().isoformat()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, count FROM activity_log WHERE user_id = ? AND date = ?",
        (user_id, today),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE activity_log SET count = ? WHERE id = ?",
            (row["count"] + 1, row["id"]),
        )
    else:
        cur.execute(
            "INSERT INTO activity_log (user_id, date, count) VALUES (?, ?, 1)",
            (user_id, today),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------
# Endpoint 1: GET /api/questions/<form_id>
# Serves the question + options for one form page.
# P1's frontend calls this once per page load.
# ---------------------------------------------------------------
@app.route("/api/questions/<form_id>", methods=["GET"])
def get_question(form_id):
    with open(QUESTIONS_PATH, "r") as f:
        all_questions = json.load(f)

    question = all_questions.get(form_id)
    if not question:
        return jsonify({"error": f"No question found for form_id {form_id}"}), 404

    return jsonify(question)


# ---------------------------------------------------------------
# Endpoint 2: POST /api/submit
# REWRITTEN per P1's contract (previously took per-form answers --
# P1's frontend now assembles all 5 forms client-side into one profile
# object before calling this).
#
# Expected JSON body:
#   {
#     "profile": { "student": { "past": {...}, "present": {...},
#                                "situation": {...}, "goal": {...},
#                                "skills": {...} } },
#     "session_id": "..."   // optional -- omit on first call
#   }
#
# Validates "profile" against schema/student_profile.schema.json,
# saves it, and returns a stable session_id the frontend must pass
# back on every subsequent call (generate-roadmap, mission, copilot).
# ---------------------------------------------------------------
@app.route("/api/submit", methods=["POST"])
def submit_profile():
    data = request.get_json(silent=True)
    if not data or "profile" not in data:
        return jsonify({"error": "Request body must include a 'profile' key"}), 400

    profile_obj = data["profile"]
    error = validate_against_schema(profile_obj, "student_profile.schema.json")
    if error:
        return jsonify({"error": error}), 400

    user_id, session_id = resolve_session(data.get("session_id"))

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO student_profiles (user_id, profile_json) VALUES (?, ?)",
        (user_id, json.dumps(profile_obj)),
    )
    conn.commit()
    profile_id = cur.lastrowid
    conn.close()

    log_activity(user_id)

    return jsonify({
        "status": "saved",
        "session_id": session_id,
        "profile_id": profile_id,
    }), 201


# ---------------------------------------------------------------
# Endpoint 3: POST /api/generate-roadmap
# REWRITTEN per P1's contract: consumes the saved profile via
# session_id, not an explicit profile_id the frontend has to track.
#
# Expected JSON body:
#   { "session_id": "..." }   // the one returned by /api/submit
#
# Looks up this session's MOST RECENTLY saved profile, prompts the
# LLM, defensively parses the response, validates it against
# roadmap_output.schema.json, and only saves if genuinely valid.
# Retries once with the error fed back to the model if the first
# attempt fails.
#
# REWRITTEN per docs/GENERATE_VALIDATE.md:
#   - validation is now tools/validate_roadmap.py (schema + business-rule
#     checks + destination-matches-profile), not schema-only
#   - max attempts raised from 2 to 3
#   - every attempt's raw output + FAIL lines are logged, so prompt
#     problems stay visible (step 6 of the doc)
#   - retry message uses the doc's exact phrasing: "Fix these problems
#     and return JSON only: <FAIL lines>"
#   - on total failure, the error shown to the client is the doc's exact
#     UI message -- never an unvalidated roadmap, never hand-repaired JSON
# ---------------------------------------------------------------
MAX_LLM_ATTEMPTS = 5


@app.route("/api/generate-roadmap", methods=["POST"])
def generate_roadmap():
    data = request.get_json(silent=True) or {}
    user_id, session_id = resolve_session(data.get("session_id"))

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, profile_json FROM student_profiles "
        "WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({
            "error": "No profile found for this session. Call /api/submit first.",
            "session_id": session_id,
        }), 404

    profile_id = row["id"]
    profile_dict = json.loads(row["profile_json"])

    retry_instruction = None
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            raw_text = call_llm_for_roadmap(profile_dict, retry_instruction=retry_instruction)
        except Exception as e:
            logger.error(f"[roadmap-gen attempt {attempt}] LLM call raised: {e}")

            if attempt < MAX_LLM_ATTEMPTS:
                wait_time = 5 * attempt
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

            retry_instruction = (
                f"Fix these problems and return JSON only: "
                f"the previous call raised an error ({e})"
            )
            continue

        logger.info(f"[roadmap-gen attempt {attempt}] raw output:\n{raw_text}")

        try:
            roadmap_dict = parse_roadmap_json(raw_text)
        except Exception as e:
            logger.info(f"[roadmap-gen attempt {attempt}] FAIL: could not parse JSON -- {e}")
            retry_instruction = f"Fix these problems and return JSON only: output was not valid JSON ({e})"
            continue

        is_valid, failures = validate_roadmap_full(roadmap_dict, profile_dict)
        if not is_valid:
            logger.info(f"[roadmap-gen attempt {attempt}] FAIL lines: {failures}")

            if attempt < MAX_LLM_ATTEMPTS:
                time.sleep(5)

            retry_instruction = (
                "Fix these problems and return JSON only: "
                + "; ".join(failures)
            )
            continue

        # Success -- save and return. Never reaches here with an
        # unvalidated roadmap, and the JSON saved is exactly what the
        # LLM produced -- never hand-repaired.
        roadmap_id = save_roadmap_to_db(profile_id, roadmap_dict, user_id=user_id)
        return jsonify({
            "status": "generated",
            "session_id": session_id,
            "roadmap_id": roadmap_id,
            "attempt": attempt,
            "destination": roadmap_dict["destination"],
            "paths": roadmap_dict["paths"],
        }), 201

    # All MAX_LLM_ATTEMPTS attempts failed -- nothing was saved.
    # Doc's exact required UI message, no internal failure details leaked.
    logger.error(
        f"Roadmap generation failed after {MAX_LLM_ATTEMPTS} attempts "
        f"for session {session_id}. Last failures: {locals().get('failures', retry_instruction)}"
    )
    return jsonify({
        "error": "Roadmap could not be generated, try again",
        "session_id": session_id,
    }), 502


# ---------------------------------------------------------------
# Endpoint: POST /api/roadmap/select-path
# Body: { "session_id": "...", "path_index": 2 }
# Student picks one of the 2-4 generated paths (0-indexed, matching
# the "paths" array /api/generate-roadmap returned). Applies to this
# session's latest roadmap. Must be called before /api/mission/current
# will work -- missions are generated from the SELECTED path, not a
# hardcoded one.
#
# NOTE (MVP limitation): once missions exist for a roadmap, re-calling
# this to pick a different path does NOT regenerate them -- that's a
# known gap, flag it if the frontend needs to support changing paths
# mid-journey.
# ---------------------------------------------------------------
@app.route("/api/roadmap/select-path", methods=["POST"])
def select_roadmap_path():
    data = request.get_json(silent=True) or {}
    path_index = data.get("path_index")
    user_id, session_id = resolve_session(data.get("session_id"))

    if path_index is None:
        return jsonify({"error": "Request body must include path_index"}), 400

    roadmap = get_latest_roadmap_for_user(user_id)
    if not roadmap:
        return jsonify({
            "error": "No roadmap found for this session. Call /api/generate-roadmap first.",
            "session_id": session_id,
        }), 404

    if not isinstance(path_index, int) or path_index < 0 or path_index >= len(roadmap["paths"]):
        return jsonify({
            "error": f"path_index must be an integer between 0 and {len(roadmap['paths']) - 1} "
                     f"for this roadmap (it has {len(roadmap['paths'])} paths)."
        }), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE roadmaps SET selected_path_index = ? WHERE id = ?",
        (path_index, roadmap["id"]),
    )
    conn.commit()
    conn.close()

    selected_path = roadmap["paths"][path_index]
    return jsonify({
        "session_id": session_id,
        "roadmap_id": roadmap["id"],
        "selected_path_index": path_index,
        "selected_path": selected_path,
    })


# ---------------------------------------------------------------
# Endpoint 4: GET /api/progress
# Returns this user's activity log, shaped for a heatmap.
# P4's frontend calls this to draw the streak graph.
# ---------------------------------------------------------------
@app.route("/api/progress", methods=["GET"])
def get_progress():
    user_id = get_session_user_id()

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT date, count FROM activity_log WHERE user_id = ? ORDER BY date",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()

    # Shape it as {date: count}, which is what most heatmap JS libs expect
    activity = {row["date"]: row["count"] for row in rows}
    return jsonify({"activity": activity})


# =================================================================
# Schema-driven endpoints (student_profile + roadmap_output)
# These are separate from the forms-based /api/submit flow above --
# once all 5 forms are filled, the frontend assembles them into one
# student_profile object matching schema/student_profile.schema.json
# and posts it here.
# =================================================================

# ---------------------------------------------------------------
# Endpoint: POST /api/profile
# Saves a full student profile. Body must match
# schema/student_profile.schema.json (top-level key: "student").
# ---------------------------------------------------------------
@app.route("/api/profile", methods=["POST"])
def save_profile():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    error = validate_against_schema(data, "student_profile.schema.json")
    if error:
        return jsonify({"error": error}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO student_profiles (profile_json) VALUES (?)",
        (json.dumps(data),),
    )
    conn.commit()
    profile_id = cur.lastrowid
    conn.close()

    return jsonify({"status": "saved", "profile_id": profile_id}), 201


# ---------------------------------------------------------------
# Endpoint: GET /api/profile/<profile_id>
# Fetches a saved profile back. Not in the original spec, but added
# so profile saves can be verified/tested the same way roadmaps can,
# and so P1's frontend can re-load a profile if needed.
# ---------------------------------------------------------------
@app.route("/api/profile/<int:profile_id>", methods=["GET"])
def get_profile(profile_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, profile_json, created_at FROM student_profiles WHERE id = ?",
        (profile_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": f"No profile found with id {profile_id}"}), 404

    return jsonify({
        "profile_id": row["id"],
        "created_at": row["created_at"],
        "profile": json.loads(row["profile_json"]),
    })


# ---------------------------------------------------------------
# Endpoint: POST /api/roadmap
# Saves a roadmap. Body must match schema/roadmap_output.schema.json
# (top-level key: "paths"), optionally with a "profile_id" field to
# link it to a saved profile. Validated BEFORE saving -- if it fails
# schema validation, nothing is written to the DB and a 400 with the
# specific validation error is returned.
# ---------------------------------------------------------------
@app.route("/api/roadmap", methods=["POST"])
def save_roadmap():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Dhruv's schema has additionalProperties: false at the root -- so
    # profile_id (which isn't part of the roadmap shape itself) must be
    # pulled out BEFORE validation, or a perfectly valid roadmap would
    # fail just for having this one extra convenience field attached.
    data = dict(data)  # don't mutate the caller's dict
    profile_id = data.pop("profile_id", None)

    error = validate_against_schema(data, "roadmap_output.schema.json")
    if error:
        return jsonify({"error": error}), 400

    roadmap_id = save_roadmap_to_db(profile_id, data)

    return jsonify({"status": "saved", "roadmap_id": roadmap_id}), 201


# ---------------------------------------------------------------
# Endpoint: GET /api/roadmap/<roadmap_id>
# Fetches a saved roadmap by its database id.
# ---------------------------------------------------------------
@app.route("/api/roadmap/<int:roadmap_id>", methods=["GET"])
def get_roadmap(roadmap_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, profile_id, generated_json, created_at FROM roadmaps WHERE id = ?",
        (roadmap_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": f"No roadmap found with id {roadmap_id}"}), 404

    roadmap_data = json.loads(row["generated_json"])
    return jsonify({
        "roadmap_id": row["id"],
        "profile_id": row["profile_id"],
        "created_at": row["created_at"],
        # .get() not [] -- roadmaps saved before the destination field
        # existed won't have it, and this endpoint shouldn't crash on them.
        "destination": roadmap_data.get("destination"),
        "paths": roadmap_data["paths"],
    })


# Create tables at import time, not just under __main__ -- gunicorn
# imports this module directly and never runs the __main__ block, so
# if init_db() only lived there, production would boot with no tables.
init_db()


# =================================================================
# Mission + Copilot endpoints
# Contract confirmed with P1 -- see README for the full spec.
# =================================================================

# ---------------------------------------------------------------
# Endpoint: GET /api/mission/current
# Query param: ?session_id=...
# Returns the student's current (earliest incomplete) mission for
# their latest roadmap's SELECTED path. Auto-generates the mission
# list from that path the first time this is called for the roadmap.
# Requires POST /api/roadmap/select-path to have been called first.
# ---------------------------------------------------------------
@app.route("/api/mission/current", methods=["GET"])
def mission_current():
    client_session_id = request.args.get("session_id")
    user_id, session_id = resolve_session(client_session_id)

    roadmap = get_latest_roadmap_for_user(user_id)
    if not roadmap:
        return jsonify({
            "error": "No roadmap found for this session. Call /api/generate-roadmap first.",
            "session_id": session_id,
        }), 404

    if roadmap["selected_path_index"] is None:
        return jsonify({
            "error": "No path selected yet. Call POST /api/roadmap/select-path first.",
            "session_id": session_id,
            "roadmap_id": roadmap["id"],
            "available_paths": len(roadmap["paths"]),
        }), 409  # 409 Conflict: roadmap exists, but isn't ready for this action yet

    selected_path = roadmap["paths"][roadmap["selected_path_index"]]
    get_or_create_missions(user_id, roadmap["id"], selected_path)
    mission_row = get_current_mission_row(user_id, roadmap["id"])

    if not mission_row:
        return jsonify({
            "session_id": session_id,
            "status": "all_complete",
            "message": "All missions for this roadmap are complete.",
        })

    return jsonify({
        "session_id": session_id,
        "status": "ok",
        "mission": mission_row_to_dict(mission_row),
    })


# ---------------------------------------------------------------
# Endpoint: POST /api/mission/complete
# Body: { "session_id": "...", "mission_id": 3 }
# Marks the given mission complete, logs activity (feeds /api/progress
# and the streak graph), and returns the next mission.
# ---------------------------------------------------------------
@app.route("/api/mission/complete", methods=["POST"])
def mission_complete():
    data = request.get_json(silent=True) or {}
    mission_id = data.get("mission_id")
    user_id, session_id = resolve_session(data.get("session_id"))

    if not mission_id:
        return jsonify({"error": "Request body must include mission_id"}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM missions WHERE id = ? AND user_id = ?", (mission_id, user_id)
    )
    mission_row = cur.fetchone()

    if not mission_row:
        conn.close()
        return jsonify({"error": f"No mission {mission_id} found for this session"}), 404

    if mission_row["status"] == "completed":
        conn.close()
        return jsonify({
            "session_id": session_id,
            "status": "already_completed",
            "mission_id": mission_id,
        })

    cur.execute(
        "UPDATE missions SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
        (mission_id,),
    )
    roadmap_id = mission_row["roadmap_id"]
    conn.commit()
    conn.close()

    log_activity(user_id)  # feeds /api/progress + the streak graph

    next_mission_row = get_current_mission_row(user_id, roadmap_id)
    if next_mission_row:
        # Promote it from "pending" to "active" now that it's the current one
        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE missions SET status = 'active' WHERE id = ?", (next_mission_row["id"],))
        conn.commit()
        conn.close()
        next_mission = mission_row_to_dict(next_mission_row)
        next_mission["status"] = "active"
    else:
        next_mission = None

    return jsonify({
        "session_id": session_id,
        "status": "completed",
        "completed_mission_id": mission_id,
        "next_mission": next_mission,  # null if that was the last one
    })


# ---------------------------------------------------------------
# Endpoint: POST /api/copilot
# Body: { "session_id": "...", "message": "Should I learn Python before SQL?" }
# Conversational assistant. Automatically pulls the student's profile,
# latest roadmap, and current mission as context -- frontend only ever
# sends session_id + the message text.
# ---------------------------------------------------------------
@app.route("/api/copilot", methods=["POST"])
def copilot():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message")
    user_id, session_id = resolve_session(data.get("session_id"))

    if not user_message:
        return jsonify({"error": "Request body must include 'message'"}), 400

    # Gather context -- profile is required, roadmap/mission are optional
    # (student might ask the copilot something before generating a roadmap).
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT profile_json FROM student_profiles WHERE user_id = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    )
    profile_row = cur.fetchone()
    conn.close()

    if not profile_row:
        return jsonify({
            "error": "No profile found for this session. Call /api/submit first.",
            "session_id": session_id,
        }), 404

    profile_dict = json.loads(profile_row["profile_json"])
    roadmap = get_latest_roadmap_for_user(user_id)
    mission_dict = None
    roadmap_context = None

    if roadmap:
        if roadmap["selected_path_index"] is not None:
            # Student has committed to one path -- that's the relevant
            # context, not all 2-4 candidate paths (which would confuse
            # the copilot about which one is actually "the plan").
            roadmap_context = {
                "selected_path": roadmap["paths"][roadmap["selected_path_index"]],
            }
            mission_row = get_current_mission_row(user_id, roadmap["id"])
            if mission_row:
                mission_dict = mission_row_to_dict(mission_row)
        else:
            # No path chosen yet -- student might be asking the copilot
            # to help them decide, so give it all the candidate paths.
            roadmap_context = {"candidate_paths_not_yet_chosen": roadmap["paths"]}

    try:
        reply_text = call_llm_for_copilot(
            profile_dict,
            roadmap_context,
            mission_dict,
            user_message,
        )
    except Exception as e:
        logger.exception("Copilot LLM call failed")
        return jsonify({"error": "Copilot is temporarily unavailable. Please try again."}), 502

    return jsonify({
        "session_id": session_id,
        "reply": reply_text,
    })

@app.route("/api/test-gemini", methods=["GET"])
def test_gemini():
    try:
        client = get_client()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly: Gemini connection successful."
        )

        return jsonify({
            "success": True,
            "model": GEMINI_MODEL,
            "response": response.text,
        }), 200

    except Exception as e:
        logger.exception("Gemini test failed")

        return jsonify({
            "success": False,
            "model": GEMINI_MODEL,
            "error": str(e),
        }), 502

if __name__ == "__main__":
    # Render/Railway inject PORT -- fall back to 5000 for local dev.
    port = int(os.environ.get("PORT", 5000))
    # Debug mode (auto-reload + interactive tracebacks) is a security
    # risk in production -- only enable it when FLASK_DEBUG=1 is set
    # explicitly (i.e. on your own machine, never on the deployed host).
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"

    app.run(debug=debug_mode, port=port, host="0.0.0.0")