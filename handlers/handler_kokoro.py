"""
Kokoro-82M 模型处理器
- 注册路由：/tts/submit, /tts/status/<task_id>, /tts/download/<task_id>, /tts/speakers
- 启动 worker 守护线程
- 懒加载 KPipeline 实例

Kokoro 特点：
- 用 voice 参数指定音色（无 speaker/instruct 概念）
- 中文支持有限（lang_code='z'）
- 采样率固定 24000 Hz
- 不支持 instruct / temperature 等采样参数（忽略，不报错）
- 同步生成，但为了接口兼容仍走 task_queue 架构
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

# ── 模型（按语言缓存）────────────────────────────────────
_models = {}          # lang_code -> KPipeline 实例
_model_lock = threading.Lock()

# ── 默认音色列表 ─────────────────────────────────────────
_SPEAKERS = [
    {"name": "af_heart",   "description": "Female, warm and heartfelt"},
    {"name": "af_alloy",   "description": "Female, balanced and versatile"},
    {"name": "af_aoede",   "description": "Female, soft and melodic"},
    {"name": "af_bella",   "description": "Female, bright and expressive"},
    {"name": "af_jessica", "description": "Female, clear and natural"},
    {"name": "af_kore",    "description": "Female, warm mid-tone"},
    {"name": "af_nova",    "description": "Female, modern and crisp"},
    {"name": "af_river",   "description": "Female, calm and flowing"},
    {"name": "af_sarah",   "description": "Female, gentle and smooth"},
    {"name": "af_sky",     "description": "Female, light and airy"},
    {"name": "am_adam",    "description": "Male, deep and steady"},
    {"name": "am_echo",    "description": "Male, resonant and rich"},
    {"name": "am_eric",    "description": "Male, warm and friendly"},
    {"name": "am_fenrir",  "description": "Male, strong and bold"},
    {"name": "am_liam",    "description": "Male, clear and youthful"},
    {"name": "am_michael", "description": "Male, smooth and professional"},
    {"name": "am_onyx",    "description": "Male, dark and powerful"},
    {"name": "am_puck",    "description": "Male, playful and energetic"},
    {"name": "am_santa",   "description": "Male, jolly and festive"},
]

# ── 语言映射 ─────────────────────────────────────────────
_LANG_MAP = {
    "Chinese": "z",
    "English": "a",
}

# ── 固定采样率 ───────────────────────────────────────────
_SAMPLE_RATE = 24000


# ══════════════════════════════════════════════════════════
# 模型加载（懒加载，按语言）
# ══════════════════════════════════════════════════════════
def _get_pipeline(lang_code: str):
    """获取指定语言的 KPipeline 实例（懒加载）"""
    if lang_code in _models:
        return _models[lang_code]

    with _model_lock:
        if lang_code in _models:
            return _models[lang_code]

        from kokoro import KPipeline
        log.info("正在加载 Kokoro 模型: lang_code=%s", lang_code)
        _models[lang_code] = KPipeline(lang_code=lang_code)
        log.info("Kokoro 模型加载完成: lang_code=%s", lang_code)
        return _models[lang_code]


def get_model(config=None):
    """
    懒加载返回 KPipeline 实例。
    config["model"].get("lang_code", "a") 决定默认语言。
    注意：KPipeline 按语言分实例，worker 中按 language 动态创建 pipeline。
    """
    lang_code = "a"
    if config is not None:
        lang_code = config.get("model", {}).get("lang_code", "a")
    return _get_pipeline(lang_code)


# ══════════════════════════════════════════════════════════
# Worker
# ══════════════════════════════════════════════════════════
def _worker_loop(config):
    """从 task_queue 取任务，调用 KPipeline 生成音频"""
    import soundfile as sf
    import numpy as np

    log.info("Worker 线程启动（Kokoro-82M），等待任务...")
    while True:
        task = _task_queue.get()
        if task is None:  # 毒丸退出
            break

        task_id = task["task_id"]
        with _store_lock:
            _task_store[task_id]["status"] = "processing"
            _task_store[task_id]["started_at"] = datetime.now().isoformat()

        try:
            text = task["text"]
            language = task.get("language", "English")
            voice = task.get("speaker", "af_heart") or "af_heart"

            # 语言 -> lang_code
            lang_code = _LANG_MAP.get(language, "a")
            if language not in _LANG_MAP:
                log.warning("[%s] 未知语言 '%s'，默认使用 English (a)", task_id[:8], language)

            # 不支持的参数仅记录日志，不报错
            if task.get("instruct"):
                log.info("[%s] Kokoro 不支持 instruct 参数，已忽略", task_id[:8])
            if task.get("temperature") is not None:
                log.info("[%s] Kokoro 不支持 temperature 参数，已忽略", task_id[:8])
            if task.get("do_sample") is not None:
                log.info("[%s] Kokoro 不支持 do_sample 参数，已忽略", task_id[:8])
            if task.get("top_k") is not None:
                log.info("[%s] Kokoro 不支持 top_k 参数，已忽略", task_id[:8])
            if task.get("top_p") is not None:
                log.info("[%s] Kokoro 不支持 top_p 参数，已忽略", task_id[:8])
            if task.get("repetition_penalty") is not None:
                log.info("[%s] Kokoro 不支持 repetition_penalty 参数，已忽略", task_id[:8])

            log.info(
                "[%s] 开始生成 | text=%.30s | voice=%s | lang=%s",
                task_id[:8], text, voice, lang_code,
            )

            # 获取对应语言的 pipeline
            pipeline = _get_pipeline(lang_code)

            # 生成音频（KPipeline 返回 generator）
            all_audio = []
            generator = pipeline(text, voice=voice, speed=1.0)
            for i, (gs, ps, audio) in enumerate(generator):
                all_audio.append(audio)

            # 拼接多段音频
            if len(all_audio) == 1:
                clip = all_audio[0]
            else:
                clip = np.concatenate(all_audio, axis=0)

            _output_base.mkdir(parents=True, exist_ok=True)
            filename = f"{task_id}.wav"
            filepath = _output_base / filename
            sf.write(str(filepath), clip, _SAMPLE_RATE)

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
    log.info("Worker 守护线程已启动（Kokoro-82M）")


# ══════════════════════════════════════════════════════════
# 路由注册
# ══════════════════════════════════════════════════════════
def register_routes(app):
    """注册 Kokoro-82M 相关路由"""

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
            "language": data.get("language", "English"),
            "speaker": data.get("speaker", "af_heart"),
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
        """获取 Kokoro 音色列表"""
        return jsonify({"speakers": _SPEAKERS})
