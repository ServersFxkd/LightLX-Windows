# Persistent state for LightLX
import json
import os

STATE_DIR = os.path.expanduser("~/.lightlx")
STATE_FILE = os.path.join(STATE_DIR, "state.json")

DEFAULTS = {
    "recent_models": [],
    "prefs": {"max_tokens": 512},
}

def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        return {
            "recent_models": list(s.get("recent_models", [])),
            "prefs": {**DEFAULTS["prefs"], **s.get("prefs", {})},
        }
    except Exception:
        return {"recent_models": [], "prefs": dict(DEFAULTS["prefs"])}

def save_state(state: dict) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def add_recent(state: dict, path: str, keep: int = 6) -> None:
    rec = [p for p in state.get("recent_models", []) if p != path]
    rec.insert(0, path)
    state["recent_models"] = rec[:keep]