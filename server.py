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
    - register_routes(app)        → register handler-specific routes only
    - start_worker(config, task_manager)
                                  → launch the worker daemon thread
    - get_model()                 → return the model instance (lazy-loaded)
"""
import importlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from queue import Queue

import yaml
from flask import Flask, jsonify, request, send_file

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
task_manager = None  # will be set in create_app()


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
            "register_routes(app), start_worker(config, task_manager), get_model()",
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
    from task_manager import TaskManager

    app = Flask(__name__)

    active_model: str = config.get("model", {}).get("active", "custom_voice")
    logger.info("Active model: %s", active_model)

    # ── 初始化 TaskManager ─────────────────────────────────────
    global task_manager
    db_path = config.get("task_manager", {}).get("db_path", "data/tasks.db")
    task_manager = TaskManager(db_path)

    # Import & initialise the active handler
    handler = import_handler(active_model)
    handler.register_routes(app)
    handler.start_worker(config, task_manager)

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
        stats = task_manager.stats()
        return jsonify({
            "queue_size": stats.get("queue_depth", task_manager.qsize()),
            "tasks_tracked": stats.get("total", 0),
        }), 200

    # ── 公共任务 API（所有 handler 通用）────────────────────────

    @app.route("/tts/submit", methods=["POST"])
    def submit_task():
        """提交 TTS 任务，返回 task_id"""
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text 不能为空"}), 400

        # extra 字段：存所有 handler 特有参数
        extra_keys = ("language", "speaker", "instruct", "temperature",
                      "do_sample", "top_k", "top_p", "repetition_penalty", "speed")
        extra = {k: data[k] for k in extra_keys if k in data}

        # 默认值
        if "language" not in extra:
            extra["language"] = data.get("language", "Chinese")
        if "speaker" not in extra:
            extra["speaker"] = data.get("speaker", "")
        if "instruct" not in extra:
            extra["instruct"] = data.get("instruct", "")

        priority = data.get("priority", 0)
        task_id = task_manager.submit(text, extra=extra, priority=priority)

        logger.info("[%s] 任务入队 (公共路由)", task_id[:8])
        return jsonify({"task_id": task_id, "position": task_manager.qsize()}), 202

    @app.route("/tts/status/<task_id>", methods=["GET"])
    def query_status(task_id):
        """查询任务状态"""
        task = task_manager.get(task_id)
        if not task:
            return jsonify({"error": "task_id 不存在"}), 404

        resp = {
            "task_id": task_id,
            "status": task["status"],
            "submitted_at": task.get("submitted_at"),
        }
        if task["status"] == "processing":
            resp["started_at"] = task.get("started_at")
        elif task["status"] == "success":
            resp["finished_at"] = task.get("finished_at")
            resp["download_url"] = f"/tts/download/{task_id}"
        elif task["status"] == "failed":
            resp["finished_at"] = task.get("finished_at")
            resp["error"] = task.get("error")

        return jsonify(resp)

    @app.route("/tts/download/<task_id>", methods=["GET"])
    def download_audio(task_id):
        """下载生成的音频文件"""
        task = task_manager.get(task_id)
        if not task:
            return jsonify({"error": "task_id 不存在"}), 404
        if task["status"] != "success":
            return jsonify({"error": f"任务状态不是 success，当前: {task['status']}"}), 404

        filepath = Path(task["file_path"])
        if not filepath.exists():
            return jsonify({"error": "音频文件不存在，可能已被清理"}), 404

        return send_file(
            str(filepath),
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"{task_id}.wav",
        )

    # ── 新增 API ───────────────────────────────────────────────

    @app.route("/tts/tasks", methods=["GET"])
    def list_tasks():
        """任务列表（分页、筛选、搜索）"""
        status_filter = request.args.get("status")
        search = request.args.get("search")
        offset = int(request.args.get("offset", 0))
        limit = int(request.args.get("limit", 20))
        result = task_manager.list(status=status_filter, search=search,
                                   offset=offset, limit=limit)
        return jsonify(result)

    @app.route("/tts/tasks/<task_id>/cancel", methods=["POST"])
    def cancel_task(task_id):
        """取消任务"""
        ok = task_manager.cancel(task_id)
        if not ok:
            task = task_manager.get(task_id)
            if not task:
                return jsonify({"error": "task_id 不存在"}), 404
            return jsonify({"error": f"任务状态 {task['status']} 不可取消"}), 400
        return jsonify({"status": "cancelled", "task_id": task_id})

    @app.route("/tts/tasks/<task_id>", methods=["DELETE"])
    def delete_task(task_id):
        """删除任务"""
        delete_file = request.args.get("delete_file", "true").lower() == "true"
        ok = task_manager.delete(task_id, delete_file=delete_file)
        if not ok:
            return jsonify({"error": "task_id 不存在"}), 404
        return jsonify({"status": "deleted", "task_id": task_id})

    @app.route("/tts/tasks/<task_id>/priority", methods=["PUT"])
    def set_priority(task_id):
        """调整优先级"""
        data = request.get_json(silent=True) or {}
        priority = data.get("priority", 0)
        ok = task_manager.set_priority(task_id, priority)
        if not ok:
            return jsonify({"error": "task_id 不存在"}), 404
        return jsonify({"task_id": task_id, "priority": priority})

    @app.route("/tts/tasks/<task_id>/retry", methods=["POST"])
    def retry_task(task_id):
        """重试失败任务"""
        ok = task_manager.retry(task_id)
        if not ok:
            task = task_manager.get(task_id)
            if not task:
                return jsonify({"error": "task_id 不存在"}), 404
            return jsonify({"error": f"任务状态 {task['status']} 不可重试"}), 400
        return jsonify({"status": "pending", "task_id": task_id})

    @app.route("/tts/cleanup", methods=["POST"])
    def cleanup():
        """批量清理旧任务"""
        data = request.get_json(silent=True) or {}
        days = data.get("days", 30)
        status_filter = data.get("status")
        delete_file = data.get("delete_file", True)
        deleted = task_manager.cleanup(days=days, status=status_filter,
                                       delete_file=delete_file)
        return jsonify({"deleted": deleted})

    @app.route("/tts/stats", methods=["GET"])
    def stats():
        """统计概览"""
        return jsonify(task_manager.stats())

    @app.route("/tts/settings", methods=["GET", "PUT"])
    def settings():
        """读取/修改 config.yaml"""
        if request.method == "GET":
            if CONFIG_PATH.exists():
                return send_file(str(CONFIG_PATH), mimetype="text/yaml")
            return jsonify({"error": "config.yaml 不存在"}), 404
        else:
            data = request.get_data(as_text=True)
            CONFIG_PATH.write_text(data, encoding="utf-8")
            return jsonify({"status": "saved"})

    # ── Web UI ─────────────────────────────────────────────────

    @app.route("/ui/")
    @app.route("/ui/<path:filename>")
    def serve_ui(filename="index.html"):
        """提供 Web UI 静态文件"""
        ui_dir = BASE_DIR / "ui"
        fp = ui_dir / filename
        if not fp.exists():
            return jsonify({"error": "not found"}), 404
        return send_file(str(fp))

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