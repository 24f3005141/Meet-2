# Meet 2 — Backend (P2's part)

## Setup (do this once)
```bash
pip install -r requirements.txt --break-system-packages
cp .env.example .env
# open .env and fill in a random FLASK_SECRET_KEY and your ANTHROPIC_API_KEY
```

## Run it
```bash
python3 app.py
```
This does two things: creates `data/meet2.db` with all 4 tables (if it doesn't exist yet),
then starts the server at http://127.0.0.1:5000

## Test it manually (before P1's frontend exists)
```bash
# Get form 1's questions
curl http://127.0.0.1:5000/api/questions/1

# Submit an answer (use -c/-b cookies.txt to keep the same session across calls)
curl -c cookies.txt -X POST http://127.0.0.1:5000/api/submit \
  -H "Content-Type: application/json" \
  -d '{"form_number": 1, "answer": {"stream": "Science (PCM)"}}'

# Generate a roadmap (currently returns FAKE data — Day 2 task is to make this real)
curl -b cookies.txt -X POST http://127.0.0.1:5000/api/generate-roadmap

# Check activity/progress data
curl -b cookies.txt http://127.0.0.1:5000/api/progress
```

## What's done (Day 1)
- [x] Flask app runs
- [x] SQLite schema (4 tables) created automatically on startup
- [x] Session handling via cookie + users table
- [x] All 4 endpoints respond (roadmap generation uses fake data for now)

## What's done (Day 2)
- [x] `/api/generate-roadmap` calls the real Claude API (`llm.py`)
- [x] Response is defensively parsed (strips ```json fences) and validated
      against `schema/roadmap_output.schema.json` before saving
- [x] One automatic retry if the LLM's output is malformed or fails validation —
      the specific error is fed back to the model so it can self-correct
- [x] Nothing is saved to the DB unless the output is genuinely valid

## What's done (Day 3)
- [x] Every request is logged (method, path, status, duration) via Python's
      `logging` module — visible in Render/Railway's log viewer in production
- [x] 404 and 405 return clean JSON instead of Flask's default HTML error page
- [x] Any unexpected crash is caught globally: full traceback goes to the
      server log, but the client only ever sees a safe, generic message —
      no stack traces or internal details leaked
- [x] `gunicorn` added as the production WSGI server (Flask's own dev server
      is not meant to be used in production — it's single-threaded and has
      no process management)
- [x] `Procfile` added so Render/Railway know how to start the app
- [x] Debug mode is OFF by default; only turns on if `FLASK_DEBUG=1` is
      explicitly set (your local `.env`, never on the deployed host)
- [x] Port is read from `$PORT` (what Render/Railway inject) with a fallback
      to 5000 for local dev

## Deploying (Render, free tier)
1. Push this repo to GitHub.
2. On [render.com](https://render.com) → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: leave blank — Render reads it from `Procfile` automatically.
5. Add environment variables in Render's dashboard (**not** in your repo):
   `FLASK_SECRET_KEY`, `ANTHROPIC_API_KEY`. Do **not** set `FLASK_DEBUG`.
6. Deploy. Render will show live logs — this is where your `[INFO] GET ... -> 200`
   request logs and any `[ERROR]` tracebacks will appear during the demo.

## ⚠️ Important caveat for the hackathon demo
SQLite (`data/meet2.db`) lives on the container's local disk. On Render/Railway's
**free tier, this disk is wiped on every redeploy and restart** — so any profiles/
roadmaps saved will disappear if the service restarts mid-demo. For a hackathon
this is usually fine (re-seed fresh demo data before judging), but don't push a
new deploy in the middle of your live demo, and know this is why — it's not a bug.

## Files
- `app.py` — the Flask app and all routes
- `database.py` — DB connection + init helper functions
- `schema.sql` — table definitions
- `questions_config.json` — form content (P5 will expand this)
- `.env.example` — copy to `.env`, never commit the real `.env`