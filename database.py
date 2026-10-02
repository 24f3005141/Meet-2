import sqlite3
import os

# The .db file will live inside data/meet2.db
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "meet2.db")

# Make sure the data/ folder exists before anything tries to open a file
# inside it. Without this, sqlite3.connect() fails with:
#   sqlite3.OperationalError: unable to open database file
# whenever this project is copied to a fresh machine that doesn't already
# have the data/ folder.
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db():
    """
    Opens a connection to the SQLite file.
    row_factory = sqlite3.Row lets us access columns by name
    (row["session_id"]) instead of by index (row[1]) — much less
    error-prone when the schema changes.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(table, column_name, column_type):
    """
    Lightweight auto-migration: adds a column to an existing table if it's
    missing, without touching existing rows or needing the DB file deleted.

    CREATE TABLE IF NOT EXISTS only helps on a brand-new DB file -- it does
    NOT retrofit new columns onto a table that already exists (this has
    bitten us three times now: user_id on student_profiles, profile_id on
    roadmaps, etc). This function is the permanent fix: call it once per
    column, after init_db(), and it's safe to call every single startup.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    existing_columns = [row["name"] for row in cur.fetchall()]

    if column_name not in existing_columns:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")
        conn.commit()
        print(f"Migrated: added column '{column_name}' to '{table}'")

    conn.close()


def init_db():
    """
    Reads schema.sql and runs it against the database.
    CREATE TABLE IF NOT EXISTS means this is safe to run every time
    the app starts — it won't wipe existing data or recreate tables
    that already exist.

    Any column added AFTER a table's first CREATE TABLE must be applied
    via ensure_column() below (not by editing the CREATE TABLE statement
    in schema.sql), or it will silently never reach already-existing
    databases.
    """
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    conn = get_db()
    with open(schema_path, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

    # --- Migrations: columns added after the original table design ---
    ensure_column("roadmaps", "selected_path_index", "INTEGER")

    print(f"Database ready at {DB_PATH}")


def get_or_create_user(session_id):
    """
    Given a session_id (from the browser cookie), find the matching
    user row, or create one if this is their first visit.
    Returns the user's integer id (the primary key), which every
    other table uses as a foreign key.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE session_id = ?", (session_id,))
    row = cur.fetchone()

    if row:
        user_id = row["id"]
    else:
        cur.execute(
            "INSERT INTO users (session_id) VALUES (?)", (session_id,)
        )
        conn.commit()
        user_id = cur.lastrowid

    conn.close()
    return user_id