"""
CustomVoice 模型处理器
- 注册路由：/tts/submit, /tts/status/<task_id>, /tts/download/<task_id>, /tts/speakers
- 启动 worker 守护线程
- 懒加载 Qwen3TTSModel 实例
"""
import uuid
import threading
import logging
from datetime import datetime
from pathlib import Path

from flask import request, jsonify, send_file

log = logging.getLogger("tts-api")

# ── 全局状态（由 start_worker 注入）──────────────────────
_task_queue = None
_task_store = None
_store_lock = None
_output_base = None

# ── 模型 ─────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()

# ── 采样保守默认值 ───────────────────────────────────────
_DEF_TEMPERATURE = 0.3
_DEF_DO_SAMPLE = True
_DEF_TOP_K = 20
_DEF_TOP_P = 0.85
_DEF_REPETITION_PENALTY = 1.1

# ── 说话人列表 ───────────────────────────────────────────
_SPEAKERS = [
    {"name": "Aiden",     "description": "阳光美声男中音，清亮通透"},
    {"name": "Dylan",     "description": "青春北京男声，清澈自然"},
    {"name": "Eric",      "description": "活泼成都男声，略带沙哑的明亮感"},
    {"name": "Ono_Anna",  "description": "俏皮日式女声，轻盈灵动"},
    {"name": "Ryan",      "description": "动感男声，节奏感强"},
    {"name": "Serena",    "description": "温柔年轻女声，暖甜细腻"},
    {"name": "Sohee",     "description": "温暖韩语女声，情感丰富"},
    {"name": "Uncle_Fu",  "description": "成熟男声，低沉醇厚"},
    {"name": "Vivian",    "description": "明亮年轻女声，略带锐利"},
]


# ══════════════════════════════════════════════════════════
# 模型加载（懒加载）
# ══════════════════════════════════════════════════════════
def get_model(config=None):
    """懒加载返回 Qwen3TTSModel 实例"""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        import torch
        from qwen_tts import Qwen3TTSModel

        if config is None:
            raise RuntimeError("config 不能为空")

        model_path = config["model_path"]
        device_map = config.get("device_map", "cpu")
        torch_dtype_str = config.get("torch_dtype", "float32")

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(torch_dtype_str, torch.float32)

        log.info("正在加载模型: %s", model_path)
        _model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        log.info("模型加载完成")
        return _model


# ══════════════════════════════════════════════════════════
# Worker
# ══════════════════════════════════════════════════════════
def _worker_loop(config):
    """从 task_queue 取任务，调用 model.generate_custom_voice() 生成音频"""
    import soundfile as sf
    import numpy as np

    log.info("Worker 线程启动，等待任务...")
    while True:
        task = _task_queue.get()
        if task is None:  # 毒丸退出
            break

        task_id = task["task_id"]
        with _store_lock:
            _task_store[task_id]["status"] = "processing"
            _task_store[task_id]["started_at"] = datetime.now().isoformat()

        try:
            model = get_model(config)

            text = task["text"]
            language = task.get("language", "Chinese")
            speaker = task.get("speaker", "")
            instruct = task.get("instruct", "")

            # 采样参数：从 task 中读取，None 则用保守默认值
            temperature = task["temperature"] if task.get("temperature") is not None else _DEF_TEMPERATURE
            do_sample    = task["do_sample"]    if task.get("do_sample")    is not None else _DEF_DO_SAMPLE
            top_k        = task["top_k"]        if task.get("top_k")        is not None else _DEF_TOP_K
            top_p        = task["top_p"]        if task.get("top_p")        is not None else _DEF_TOP_P
            rep_penalty  = task["repetition_penalty"] if task.get("repetition_penalty") is not None else _DEF_REPETITION_PENALTY

            generate_kwargs = {
                "temperature": temperature,
                "do_sample": do_sample,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": rep_penalty,
            }

            log.info(
                "[%s] 开始生成 | text=%.30s | speaker=%s | instruct=%s | "
                "temp=%.2f | sample=%s | top_k=%d | top_p=%.2f | rep_pen=%.2f",
                task_id[:8], text, speaker, instruct,
                temperature, do_sample, top_k, top_p, rep_penalty,
            )

            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                **generate_kwargs,
            )

            clip = wavs[0] if isinstance(wavs, list) else wavs

            _output_base.mkdir(parents=True, exist_ok=True)
            filename = f"{task_id}.wav"
            filepath = _output_base / filename
            sf.write(str(filepath), clip, sr)

            with _store_lock:
                _task_store[task_id]["status"] = "success"
                _task_store[task_id]["file_path"] = str(filepath)
                _task_store[task_id]["finished_at"] = datetime.now().isoformat()

            log.info("[%s] 生成成功: %s", task_id[:8], filepath)

        except Exception as e:
            log.error("[%s] 生成失败: %s", task_id[:8], e, exc_info=True)
            with _store_lock:
                _task_store[task_id]["status"] = "failed"
                _task_store[task_id]["error"] = str(e)
                _task_store[task_id]["finished_at"] = datetime.now().isoformat()

        finally:
            _task_queue.task_done()


def start_worker(config, task_queue, task_store, store_lock):
    """启动 worker 守护线程"""
    global _task_queue, _task_store, _store_lock, _output_base

    _task_queue = task_queue
    _task_store = task_store
    _store_lock = store_lock
    _output_base = Path(config.get("output_base_dir") or config.get("output", {}).get("base_dir", "output_audio"))

    worker_thread = threading.Thread(target=_worker_loop, args=(config,), daemon=True)
    worker_thread.start()
    log.info("Worker 守护线程已启动")


# ══════════════════════════════════════════════════════════
# 路由注册
# ══════════════════════════════════════════════════════════
def register_routes(app):
    """注册 CustomVoice 相关路由"""

    @app.route("/tts/submit", methods=["POST"])
    def submit_task():
        """提交 TTS 任务，返回 task_id"""
        data = request.get_json(silent=True) or {}

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text 不能为空"}), 400

        date_prefix = datetime.now().strftime("%Y%m%d")
        task_id = f"{date_prefix}_{uuid.uuid4().hex}"
        task = {
            "task_id": task_id,
            "text": text,
            "language": data.get("language", "Chinese"),
            "speaker": data.get("speaker", ""),
            "instruct": data.get("instruct", ""),
            "temperature": data.get("temperature"),
            "do_sample": data.get("do_sample"),
            "top_k": data.get("top_k"),
            "top_p": data.get("top_p"),
            "repetition_penalty": data.get("repetition_penalty"),
            "status": "pending",
            "submitted_at": datetime.now().isoformat(),
        }

        with _store_lock:
            _task_store[task_id] = task

        _task_queue.put(task)
        pos = _task_queue.qsize()

        log.info("[%s] 任务入队，当前排队: %d", task_id[:8], pos)
        return jsonify({"task_id": task_id, "position": pos}), 202

    @app.route("/tts/status/<task_id>", methods=["GET"])
    def query_status(task_id):
        """查询任务状态"""
        with _store_lock:
            task = _task_store.get(task_id)
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
        with _store_lock:
            task = _task_store.get(task_id)
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

    @app.route("/tts/speakers", methods=["GET"])
    def list_speakers():
        """获取说话人列表"""
        return jsonify({"speakers": _SPEAKERS})
