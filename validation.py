import json
import os
from jsonschema import validate, ValidationError

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schema")


def load_schema(filename):
    """Reads one schema file from the schema/ folder."""
    path = os.path.join(SCHEMA_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


def validate_against_schema(data, schema_filename):
    """
    Validates `data` (a Python dict, already parsed from JSON) against
    the schema in schema/<schema_filename>.

    Returns None if valid.
    Returns a human-readable error string if invalid -- this is what
    gets sent back to the client in the 400 response, so it tells
    whoever is calling the API exactly what's wrong with their payload.
    """
    schema = load_schema(schema_filename)
    try:
        validate(instance=data, schema=schema)
        return None
    except ValidationError as e:
        # e.path is the location inside the JSON where it failed,
        # e.g. ["paths", 0, "roadmap_steps"] -> "paths/0/roadmap_steps"
        location = "/".join(str(p) for p in e.path) or "(root)"
        return f"Validation failed at '{location}': {e.message}"
