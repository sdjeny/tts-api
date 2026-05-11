"""
Qwen3-TTS API 客户端
供第三方调用，内部统一处理 User-Agent、超时、重试等细节

用法：
    from tts_client import TtsClient

    client = TtsClient("https://your-server.com")
    task_id = client.submit(text="你好世界", speaker="Dylan", instruct="愉快")
    result = client.wait(task_id)
    if result.ok:
        data = client.download(task_id)
        with open("out.wav", "wb") as f:
            f.write(data)
"""
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────
_DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_DEFAULT_TIMEOUT = 120  # 秒
_DEFAULT_POLL_INTERVAL = 5  # 秒
_DEFAULT_MAX_WAIT = 600  # 秒


# ──────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────
@dataclass
class SubmitResult:
    task_id: str = ""
    position: int = 0
    raw: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class StatusResult:
    task_id: str = ""
    status: str = ""          # pending / processing / success / failed
    submitted_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    download_url: str = ""
    error: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.status == "success"

    @property
    def done(self):
        return self.status in ("success", "failed")


@dataclass
class DownloadResult:
    data: bytes = b""
    status_code: int = 0
    error: str = ""

    @property
    def ok(self):
        return self.status_code == 200 and len(self.data) > 0


# ──────────────────────────────────────────────────────
# 客户端
# ──────────────────────────────────────────────────────
class TtsClient:
    def __init__(self, base_url: str = None, timeout: int = _DEFAULT_TIMEOUT,
                 poll_interval: int = _DEFAULT_POLL_INTERVAL,
                 max_wait: int = _DEFAULT_MAX_WAIT,
                 user_agent: str = _DEFAULT_UA,
                 config_path: str = None):
        """
        参数：
            base_url: 服务地址，不传则自动从 config.yaml 读取
            timeout: 请求超时（秒）
            poll_interval: 轮询间隔（秒）
            max_wait: 最大等待时间（秒）
            user_agent: User-Agent
            config_path: config.yaml 路径，不传则自动从模块同级目录查找
        """
        if not base_url:
            base_url = self._load_base_url(config_path)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self._ua = user_agent

    @classmethod
    def from_config(cls, config_path: str = None, **kwargs) -> "TtsClient":
        """
        从 config.yaml 创建客户端，一行搞定：
            client = TtsClient.from_config()
        """
        return cls(base_url=None, config_path=config_path, **kwargs)

    @staticmethod
    def _load_base_url(config_path: str = None) -> str:
        """从 config.yaml 读取 api.base_url"""
        if config_path is None:
            # 默认：模块文件所在目录的 config.yaml
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml 不存在: {config_path}")

        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        url = cfg.get("api", {}).get("base_url", "")
        if not url:
            raise ValueError(f"config.yaml 中未配置 api.base_url: {config_path}")
        return url

    # ── 内部请求 ──────────────────────────────────────
    def _request(self, method: str, path: str, data: dict = None):
        url = self.base_url + path
        body = json.dumps(data).encode("utf-8") if data else None
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self._ua,
        }
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                try:
                    return json.loads(raw), resp.status
                except (json.JSONDecodeError, ValueError):
                    return {"_raw": raw}, resp.status
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return json.loads(raw), e.code
            except (json.JSONDecodeError, ValueError):
                return {"_raw": raw, "_http_error": e.code}, e.code

    def _download(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": self._ua})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read(), resp.status

    # ── 提交任务 ──────────────────────────────────────
    def submit(self, text: str, language: str = "zh",
               speaker: str = "", instruct: str = "",
               temperature: float = None, do_sample: bool = None,
               top_k: int = None, top_p: float = None,
               repetition_penalty: float = None,
               speed: float = None) -> SubmitResult:
        """
        提交 TTS 任务。

        采样控制参数（可选，None 则服务端使用保守默认值）：
          temperature:        采样温度，越低声音越稳定，建议 0.1~1.0
          do_sample:          是否采样，True=采样 / False=贪心解码
          top_k:              保留 top-k token 采样，越小越集中，建议 10~100
          top_p:              核采样阈值，越小越集中，建议 0.5~1.0
          repetition_penalty: 重复惩罚，>1.0 抑制重复，建议 1.0~1.5
          speed:              语速，0.25~4.0，默认 1.0（仅 Kokoro 支持）
        """
        payload = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": instruct,
        }
        # 仅当客户端显式传入时才携带采样参数，避免旧调用意外覆盖服务端默认值
        if temperature is not None:
            payload["temperature"] = temperature
        if do_sample is not None:
            payload["do_sample"] = do_sample
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if speed is not None:
            payload["speed"] = speed

        r, code = self._request("POST", "/tts/submit", payload)
        if code == 202:
            return SubmitResult(task_id=r["task_id"], position=r.get("position", 0), raw=r)
        return SubmitResult(error=r.get("error", f"HTTP {code}"), raw=r)

    # ── 查询状态 ──────────────────────────────────────
    def status(self, task_id: str) -> StatusResult:
        r, code = self._request("GET", f"/tts/status/{task_id}")
        if code == 200:
            return StatusResult(
                task_id=task_id,
                status=r.get("status", ""),
                submitted_at=r.get("submitted_at", ""),
                started_at=r.get("started_at", ""),
                finished_at=r.get("finished_at", ""),
                download_url=r.get("download_url", ""),
                error=r.get("error", ""),
                raw=r,
            )
        return StatusResult(task_id=task_id, status="error",
                            error=r.get("error", f"HTTP {code}"), raw=r)

    # ── 下载音频 ──────────────────────────────────────
    def download(self, task_id: str) -> DownloadResult:
        sr = self.status(task_id)
        if not sr.ok:
            return DownloadResult(error=f"任务未完成: {sr.status}", status_code=0)
        url = self.base_url + sr.download_url
        try:
            data, code = self._download(url)
            return DownloadResult(data=data, status_code=code)
        except urllib.error.HTTPError as e:
            return DownloadResult(error=str(e), status_code=e.code)
        except Exception as e:
            return DownloadResult(error=str(e), status_code=0)

    # ── 等待完成 ──────────────────────────────────────
    def wait(self, task_id: str, poll_interval: int = None,
             max_wait: int = None) -> StatusResult:
        interval = poll_interval or self.poll_interval
        deadline = time.time() + (max_wait or self.max_wait)
        while time.time() < deadline:
            sr = self.status(task_id)
            if sr.done:
                return sr
            time.sleep(interval)
        return StatusResult(task_id=task_id, status="timeout", error="等待超时")

    # ── 队列概况 ──────────────────────────────────────
    def queue_info(self) -> dict:
        r, code = self._request("GET", "/tts/queue")
        if code == 200:
            return r
        return {"error": r.get("error", f"HTTP {code}")}

    # ── 健康检查 ──────────────────────────────────────
    def health(self) -> bool:
        r, code = self._request("GET", "/tts/health")
        return code == 200 and r.get("status") == "ok"
