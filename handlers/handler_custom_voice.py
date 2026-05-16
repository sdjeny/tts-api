"""
CustomVoice 模型处理器
- 注册路由：/tts/speakers（handler 特有路由）
- 启动 worker 守护线程
- 懒加载 Qwen3TTSModel 实例

注意：submit/status/download 路由已统一在 server.py 注册。
"""
import uuid
import threading
import logging
from datetime import datetime
from pathlib import Path

from flask import request, jsonify, send_file

log = logging.getLogger("tts-api")

# ── 全局状态（由 start_worker 注入）──────────────────────
_task_manager = None
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
    {"name": "aiden",     "description": "阳光美声男中音，清亮通透"},
    {"name": "dylan",     "description": "青春北京男声，清澈自然"},
    {"name": "eric",      "description": "活泼成都男声，略带沙哑的明亮感"},
    {"name": "ono_anna",  "description": "俏皮日式女声，轻盈灵动"},
    {"name": "ryan",      "description": "动感男声，节奏感强"},
    {"name": "serena",    "description": "温柔年轻女声，暖甜细腻"},
    {"name": "sohee",     "description": "温暖韩语女声，情感丰富"},
    {"name": "uncle_fu",  "description": "成熟男声，低沉醇厚"},
    {"name": "vivian",    "description": "明亮年轻女声，略带锐利"},
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

        model_path = config["model"]["model_path"]
        device_map = config.get("model", {}).get("device_map", "cpu")
        torch_dtype_str = config.get("model", {}).get("torch_dtype", "float32")

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
    """从 task_manager 取任务，调用 model.generate_custom_voice() 生成音频"""
    import soundfile as sf
    import numpy as np

    log.info("Worker 线程启动，等待任务...")
    while True:
        task = _task_manager.get_pending_task()
        if task is None:  # 毒丸退出
            break

        task_id = task["task_id"]
        _task_manager.update_status(task_id, "processing",
                                    started_at=datetime.now().isoformat())

        try:
            model = get_model(config)

            text = task["text"]
            extra = task.get("extra", {})
            language = extra.get("language", "Chinese")
            speaker = extra.get("speaker", "")
            instruct = extra.get("instruct", "")

            # 采样参数：从 extra 中读取，None 则用保守默认值
            temperature = extra.get("temperature") if extra.get("temperature") is not None else _DEF_TEMPERATURE
            do_sample    = extra.get("do_sample")    if extra.get("do_sample")    is not None else _DEF_DO_SAMPLE
            top_k        = extra.get("top_k")        if extra.get("top_k")        is not None else _DEF_TOP_K
            top_p        = extra.get("top_p")        if extra.get("top_p")        is not None else _DEF_TOP_P
            rep_penalty  = extra.get("repetition_penalty") if extra.get("repetition_penalty") is not None else _DEF_REPETITION_PENALTY

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

            _task_manager.update_status(
                task_id, "success",
                file_path=str(filepath),
                finished_at=datetime.now().isoformat(),
            )

            log.info("[%s] 生成成功: %s", task_id[:8], filepath)

        except Exception as e:
            log.error("[%s] 生成失败: %s", task_id[:8], e, exc_info=True)
            _task_manager.update_status(
                task_id, "failed",
                error=str(e),
                finished_at=datetime.now().isoformat(),
            )

        finally:
            _task_manager.mark_task_done()


def start_worker(config, task_manager):
    """启动 worker 守护线程"""
    global _task_manager, _output_base

    _task_manager = task_manager
    _output_base = Path(config.get("output_base_dir") or config.get("output", {}).get("base_dir", "output_audio"))

    worker_thread = threading.Thread(target=_worker_loop, args=(config,), daemon=True)
    worker_thread.start()
    log.info("Worker 守护线程已启动")


# ══════════════════════════════════════════════════════════
# 路由注册（仅 handler 特有路由）
# ══════════════════════════════════════════════════════════
def register_routes(app):
    """注册 CustomVoice 特有路由（speakers 列表）"""

    @app.route("/tts/speakers", methods=["GET"])
    def list_speakers():
        """获取说话人列表"""
        return jsonify({"speakers": _SPEAKERS})