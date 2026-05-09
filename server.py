#!/usr/bin/env python3
"""
TTS API — Unified Entry Point
==============================
Loads config.yaml, dynamically imports the active model handler, and starts
a Flask server with a background worker thread.

Expected config.yaml structure:
    model:
      active: custom_voice  # custom_voice | base | kokoro
    server:
      host: 0.0.0.0
      port: 8420
    output:
      base_dir: ./output_audio

Each handler module (under handlers/) must expose:
    - register_routes(app)        → register all /tts/* blueprint routes
    - start_worker(config, task_queue, task_store, store_lock)
                                  → launch the worker daemon thread
    - get_model()                 → return the model instance (lazy-loaded)
"""

import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from queue import Queue

import yaml
from flask import Flask, jsonify

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tts-api")

# ---------------------------------------------------------------------------
# Global shared state
# ---------------------------------------------------------------------------
task_queue: Queue = Queue()
task_store: dict = {}
store_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    """Load and return the YAML configuration file."""
    if not path.exists():
        logger.error("Configuration file not found: %s", path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        logger.error("config.yaml must contain a top-level mapping")
        sys.exit(1)
    return cfg


def import_handler(active: str):
    """
    Dynamically import the handler module for *active* model.

    Module name mapping:
        custom_voice → handlers.handler_custom_voice
        base        → handlers.handler_base
        kokoro      → handlers.handler_kokoro

    Returns the imported module, which must expose:
        register_routes, start_worker, get_model
    """
    name_map = {
        "custom_voice": "handler_custom_voice",
        "base": "handler_base",
        "kokoro": "handler_kokoro",
    }
    file_name = name_map.get(active, active)
    module_name = f"handlers.{file_name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        logger.error(
            "Failed to import handler module '%s' for active model '%s'.\n"
            "Make sure handlers/%s.py exists and exposes: "
            "register_routes(app), start_worker(config, task_queue, task_store, store_lock), get_model()",
            module_name,
            active,
            file_name,
        )
        raise ImportError(
            f"Handler module '{module_name}' not found. "
            f"Expected file: handlers/{file_name}.py"
        ) from exc

    # Validate required interface
    for attr in ("register_routes", "start_worker", "get_model"):
        if not hasattr(module, attr):
            raise ImportError(
                f"Handler module '{module_name}' is missing required "
                f"attribute '{attr}'. Every handler must expose: "
                f"register_routes, start_worker, get_model"
            )

    return module


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(config: dict) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    active_model: str = config.get("model", {}).get("active", "custom_voice")
    logger.info("Active model: %s", active_model)

    # Import & initialise the active handler
    handler = import_handler(active_model)
    handler.register_routes(app)
    handler.start_worker(config, task_queue, task_store, store_lock)

    # ------------------------------------------------------------------
    # Generic routes (always available regardless of active model)
    # ------------------------------------------------------------------

    @app.route("/tts/health", methods=["GET"])
    def health():
        """Health-check endpoint."""
        return jsonify({"status": "ok", "model": active_model}), 200

    @app.route("/tts/queue", methods=["GET"])
    def queue_status():
        """Return current queue size and number of tracked tasks."""
        with store_lock:
            return jsonify({
                "queue_size": task_queue.qsize(),
                "tasks_tracked": len(task_store),
            }), 200

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = load_config(CONFIG_PATH)
    app = create_app(config)

    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 8420)

    logger.info("Starting TTS API on %s:%d  [model=%s]",
                host, port, config.get("model", {}).get("active"))
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
