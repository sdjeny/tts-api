"""
Base 模型（Voice Clone）处理器
- 注册路由：TTS 任务提交/查询/下载 + Voice Clone 管理
- 启动 worker 守护线程
- 懒加载 Qwen3TTSModel 实例

Voice Clone 存储结构（voice_clones/<name>/）：
  ├── meta.json        { name, description, ref_text, created_at }
  ├── ref_audio.wav    参考音频
  └── embedding.pt     VoiceClonePromptItem 序列化
"""
import uuid
import json
import shutil
import base64
import threading
import logging
from datetime import datetime
from pathlib import Path

from flask import request, jsonify, send_file, Response

log = logging.getLogger("tts-api")

# ── 全局状态（由 start_worker 注入）──────────────────────
_task_queue = None
_task_store = None
_store_lock = None
_output_base = None
_clone_dir = None

# ── 模型 ─────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()

# ── 采样保守默认值 ───────────────────────────────────────
_DEF_TEMPERATURE = 0.3
_DEF_DO_SAMPLE = True
_DEF_TOP_K = 20
_DEF_TOP_P = 0.85
_DEF_REPETITION_PENALTY = 1.1


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

        model_path = config["model"]["model_path"]
        device_map = config.get("model", {}).get("device_map", "cpu")
        torch_dtype_str = config.get("model", {}).get("torch_dtype", "float32")

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(torch_dtype_str, torch.float32)

        log.info("正在加载 Base 模型: %s", model_path)
        _model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        log.info("Base 模型加载完成")
        return _model


# ══════════════════════════════════════════════════════════
# Voice Clone 工具函数
# ══════════════════════════════════════════════════════════
def _clone_dir(name: str) -> Path:
    """获取说话人存储目录"""
    return _clone_dir_base / name


def _clone_pt_path(name: str) -> Path:
    return _clone_dir(name) / "embedding.pt"


def _clone_audio_path(name: str) -> Path:
    return _clone_dir(name) / "ref_audio.wav"


def list_voice_clones() -> list:
    """列出所有已保存的 voice clone"""
    clones = []
    base = _clone_dir_base
    if not base.exists():
        return clones
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        pt_path = d / "embedding.pt"
        audio_path = d / "ref_audio.wav"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
        clones.append({
            "name": d.name,
            "description": meta.get("description", ""),
            "ref_text": meta.get("ref_text", ""),
            "created_at": meta.get("created_at", ""),
            "has_audio": audio_path.exists(),
            "has_pt": pt_path.exists(),
            "pt_size_bytes": pt_path.stat().st_size if pt_path.exists() else 0,
        })
    return clones


def load_voice_clone_prompt(name: str):
    """
    加载 voice clone，返回 VoiceClonePromptItem 列表。
    兼容旧格式（单个值而非列表）。
    """
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    pt_path = _clone_pt_path(name)
    if not pt_path.exists():
        return None
    import torch
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    # 从 meta.json 读取 ref_text
    meta_path = _clone_dir(name) / "meta.json"
    ref_text = ""
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                ref_text = json.load(f).get("ref_text", "")
        except Exception:
            pass

    # 兼容旧格式：单个 bool/Tensor → 包装为单元素列表
    xvec = data.get("x_vector_only_mode", False)
    icl = data.get("icl_mode", True)
    ref_code = data.get("ref_code")
    ref_spk = data.get("ref_spk_embedding")

    if not isinstance(xvec, list):
        xvec = [xvec]
    if not isinstance(icl, list):
        icl = [icl]
    if not isinstance(ref_code, (list, tuple)):
        ref_code = [ref_code]
    if not isinstance(ref_spk, (list, tuple)):
        ref_spk = [ref_spk]

    # 构造 VoiceClonePromptItem 列表
    items = []
    for i in range(len(xvec)):
        items.append(VoiceClonePromptItem(
            ref_code=ref_code[i],
            ref_spk_embedding=ref_spk[i],
            x_vector_only_mode=xvec[i],
            icl_mode=icl[i],
            ref_text=ref_text if ref_text else None,
        ))
    return items


def save_voice_clone(name: str, prompt_item, audio_data, audio_sr,
                     ref_text: str = "", description: str = ""):
    """
    保存 voice clone：
    - meta.json: { name, description, ref_text, created_at }
    - ref_audio.wav: 参考音频
    - embedding.pt: VoiceClonePromptItem 序列化
    """
    import torch
    import soundfile as sf

    d = _clone_dir(name)
    d.mkdir(parents=True, exist_ok=True)

    # 1. 保存参考音频
    audio_path = d / "ref_audio.wav"
    sf.write(str(audio_path), audio_data, audio_sr)

    # 2. 保存 embedding pt（单元素列表格式）
    save_data = {
        "ref_code": [prompt_item.ref_code],
        "ref_spk_embedding": [prompt_item.ref_spk_embedding],
        "x_vector_only_mode": [prompt_item.x_vector_only_mode],
        "icl_mode": [prompt_item.icl_mode],
    }
    pt_path = d / "embedding.pt"
    torch.save(save_data, str(pt_path))

    # 3. 保存元数据
    meta = {
        "name": name,
        "description": description,
        "ref_text": ref_text or (prompt_item.ref_text or ""),
        "created_at": datetime.now().isoformat(),
    }
    meta_path = d / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════
# Worker
# ══════════════════════════════════════════════════════════
def _worker_loop(config):
    """从 task_queue 取任务，调用 model.generate_voice_clone() 生成音频"""
    import soundfile as sf

    log.info("Worker 线程启动（Base 模型），等待任务...")
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

            # 采样参数
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

            # 加载 voice clone prompt
            prompt_items = None
            if speaker:
                prompt_items = load_voice_clone_prompt(speaker)
                if prompt_items is not None:
                    log.info("[%s] 已加载 voice clone: %s (%d items)", task_id[:8], speaker, len(prompt_items))
                else:
                    log.warning("[%s] voice clone '%s' 未找到", task_id[:8], speaker)

            log.info(
                "[%s] 开始生成 | text=%.30s | speaker=%s | instruct=%s | "
                "temp=%.2f | sample=%s | top_k=%d | top_p=%.2f | rep_pen=%.2f",
                task_id[:8], text, speaker, instruct,
                temperature, do_sample, top_k, top_p, rep_penalty,
            )

            if prompt_items is not None:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt_items,
                    non_streaming_mode=True,
                    **generate_kwargs,
                )
            else:
                raise ValueError(
                    f"speaker '{speaker}' 对应的 voice clone 不存在。"
                    f"请先在管理页面创建 voice clone，或提供有效的 speaker 名称。"
                    f"可用的 voice clone: {[c['name'] for c in list_voice_clones()]}"
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
    global _clone_dir_base

    _task_queue = task_queue
    _task_store = task_store
    _store_lock = store_lock
    _output_base = Path(config.get("output_base_dir") or config.get("output", {}).get("base_dir", "output_audio"))
    _clone_dir_base = Path(config.get("clone_dir") or config.get("voice_clones", {}).get("dir", "voice_clones"))
    _clone_dir_base.mkdir(parents=True, exist_ok=True)

    worker_thread = threading.Thread(target=_worker_loop, args=(config,), daemon=True)
    worker_thread.start()
    log.info("Worker 守护线程已启动（Base 模型）")


# ══════════════════════════════════════════════════════════
# 路由注册
# ══════════════════════════════════════════════════════════
def register_routes(app):
    """注册 Base 模型（Voice Clone）相关路由"""

    # ── TTS 任务接口 ──────────────────────────────────────

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

    # ── 说话人 / Voice Clone 列表 ─────────────────────────

    @app.route("/tts/speakers", methods=["GET"])
    def list_speakers():
        """获取说话人列表（name + description）"""
        clones = list_voice_clones()
        speakers = [{"name": c["name"], "description": c.get("description", "")} for c in clones]
        return jsonify({"speakers": speakers})

    @app.route("/tts/clones", methods=["GET"])
    def list_clones():
        """列出所有 voice clone（含 meta 信息）"""
        clones = list_voice_clones()
        return jsonify({"clones": clones})

    # ── Voice Clone CRUD ───────────────────────────────────

    @app.route("/tts/clones", methods=["POST"])
    def create_clone():
        """
        从参考音频创建 voice clone。
        支持两种输入方式：
        1. multipart/form-data: 上传音频文件 (field: audio) + name + instruct(即 ref_text)
        2. JSON: { "name": "Aiden", "audio_base64": "...", "instruct": "参考文本" }
        """
        name = ""
        audio_data = None
        audio_sr = None
        ref_text = ""
        description = ""
        x_vector_only = False

        ct = request.content_type or ""
        log.info("[create_clone] content_type=%s, content_length=%s", ct, request.content_length)

        if "multipart/form-data" in ct:
            name = (request.form.get("name") or "").strip()
            ref_text = (request.form.get("instruct") or "").strip()
            description = (request.form.get("description") or "").strip()
            x_vector_only = (request.form.get("x_vector_only") or "false").lower() == "true"

            log.info("[create_clone] form: name=%s, ref_text=%s, has_audio=%s",
                     name, ref_text, "audio" in request.files)

            if "audio" not in request.files:
                log.warning("[create_clone] 400: 未找到 audio 字段，files=%s",
                            list(request.files.keys()))
                return jsonify({"error": "请上传音频文件（field: audio）"}), 400
            audio_file = request.files["audio"]
            if not audio_file.filename:
                log.warning("[create_clone] 400: 音频文件名为空")
                return jsonify({"error": "音频文件为空"}), 400

            import soundfile as sf
            import io
            try:
                raw_bytes = audio_file.read()
                log.info("[create_clone] 音频文件大小: %d bytes", len(raw_bytes))
                data, sr = sf.read(io.BytesIO(raw_bytes))
                audio_data = data
                audio_sr = sr
                log.info("[create_clone] 音频读取成功: sr=%d, shape=%s", sr, data.shape)
            except Exception as e:
                log.error("[create_clone] 400: 音频读取失败: %s", e)
                return jsonify({"error": f"无法读取音频文件: {e}"}), 400
        else:
            data = request.get_json(silent=True) or {}
            name = (data.get("name") or "").strip()
            ref_text = (data.get("instruct") or "").strip()
            description = (data.get("description") or "").strip()
            x_vector_only = data.get("x_vector_only", False)

            audio_b64 = data.get("audio_base64", "")
            if not audio_b64:
                log.warning("[create_clone] 400: 非 multipart 且无 audio_base64")
                return jsonify({"error": "请提供 audio_base64 或上传音频文件"}), 400

            import soundfile as sf
            import io
            try:
                raw = base64.b64decode(audio_b64)
                audio_data, audio_sr = sf.read(io.BytesIO(raw))
            except Exception as e:
                log.error("[create_clone] 400: base64 解码失败: %s", e)
                return jsonify({"error": f"无法解码音频: {e}"}), 400

        if not name:
            log.warning("[create_clone] 400: name 为空")
            return jsonify({"error": "name 不能为空"}), 400
        if audio_data is None:
            log.warning("[create_clone] 400: audio_data 为空")
            return jsonify({"error": "音频数据为空"}), 400

        # x_vector_only=False 时 ref_text 必填
        if not x_vector_only and not ref_text:
            log.warning("[create_clone] 400: 未填参考文本且未开启 x_vector_only 模式")
            return jsonify({"error": "请填写参考文本，或勾选「仅使用说话人向量」模式"}), 400

        # 检查是否已存在
        if _clone_dir(name).exists():
            return jsonify({"error": f"voice clone '{name}' 已存在，请先删除或使用其他名称"}), 409

        try:
            model = get_model()
            ref_audio_input = (audio_data, audio_sr)

            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_audio_input,
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=x_vector_only,
            )

            if not prompt_items:
                return jsonify({"error": "无法从参考音频提取 voice prompt"}), 500

            save_voice_clone(
                name, prompt_items[0],
                audio_data=audio_data, audio_sr=audio_sr,
                ref_text=ref_text, description=description,
            )

            log.info("Voice clone 创建成功: %s (x_vector_only=%s)", name, x_vector_only)

            return jsonify({
                "ok": True,
                "name": name,
                "x_vector_only": x_vector_only,
                "has_ref_text": bool(ref_text),
            }), 201

        except Exception as e:
            log.error("创建 voice clone 失败: %s", e, exc_info=True)
            # 清理可能的部分写入
            d = _clone_dir(name)
            if d.exists():
                shutil.rmtree(str(d))
            return jsonify({"error": f"创建 voice clone 失败: {e}"}), 500

    @app.route("/tts/clones/<name>", methods=["DELETE"])
    def delete_clone(name):
        """删除 voice clone（整个目录）"""
        d = _clone_dir(name)
        if not d.exists():
            return jsonify({"error": f"voice clone '{name}' 不存在"}), 404
        shutil.rmtree(str(d))
        log.info("Voice clone 已删除: %s", name)
        return jsonify({"ok": True, "deleted": name})

    @app.route("/tts/clones/<name>/download", methods=["GET"])
    def download_clone_pt(name):
        """下载 voice clone 的 embedding.pt 文件"""
        pt_path = _clone_pt_path(name)
        if not pt_path.exists():
            return jsonify({"error": f"voice clone '{name}' 不存在"}), 404
        return send_file(
            str(pt_path),
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name=f"{name}_embedding.pt",
        )

    @app.route("/tts/clones/<name>/audio", methods=["GET"])
    def download_clone_audio(name):
        """获取参考音频"""
        audio_path = _clone_audio_path(name)
        if not audio_path.exists():
            return jsonify({"error": f"voice clone '{name}' 的参考音频不存在"}), 404
        return send_file(str(audio_path), mimetype="audio/wav")
