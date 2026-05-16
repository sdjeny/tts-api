# TTS API

支持三种模型的统一 TTS 服务接口，通过 `config.yaml` 切换模型，对外 API 保持一致。

## 支持的模型

| active 值 | 模型 | 特点 |
|-----------|------|------|
| `custom_voice` | Qwen3-TTS CustomVoice | 预设音色（9个），甜美女声等 |
| `base` | Qwen3-TTS Base | Voice Clone，需先注册自定义音色 |
| `kokoro` | Kokoro-82M | 轻量级，中英双语，18个音色 |

## 快速开始

### 1. 配置

```bash
cp config.yaml config.local.yaml
# 修改 config.local.yaml 中的 model_path 等配置
```

### 2. 启动

```bash
docker compose up -d
```

### 3. 验证

```bash
curl http://localhost:8420/tts/health
```

## API 接口

> 所有接口的 base URL 默认为 `http://localhost:8420`

### 健康检查
```
GET /tts/health
```
返回：`{"status": "ok", "model": "custom_voice"}`

### 提交任务
```
POST /tts/submit
Content-Type: application/json

{
  "text": "你好世界",
  "language": "Chinese",
  "speaker": "dylan",
  "instruct": "愉快，轻松",
  "temperature": 0.3,
  "top_k": 20,
  "top_p": 0.85,
  "repetition_penalty": 1.1
}
```
返回：`{"task_id": "20260509_abc123", "position": 1}` (HTTP 202)

### 查询状态
```
GET /tts/status/<task_id>
```
返回：`{"task_id": "...", "status": "success", "download_url": "/tts/download/..."}`

### 下载音频
```
GET /tts/download/<task_id>
```
返回：WAV 音频文件

### 音色列表
```
GET /tts/speakers
```
返回：`{"speakers": [{"name": "dylan", "description": "..."}]}`

### 队列状态
```
GET /tts/queue
```
返回：`{"queue_size": 0, "tasks_tracked": 5}`

## Voice Clone（Base 模型）

当 `active: base` 时，需先注册音色：

```
POST /tts/clones
Content-Type: multipart/form-data

name: my_voice
instruct: 参考音频对应的文字
audio: <wav文件>
```

已注册音色列表：
```
GET /tts/clones
```

## 采样参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `temperature` | 0.3 | 0.1~1.0 | 越低越稳定 |
| `do_sample` | true | true/false | true=采样，false=贪心解码 |
| `top_k` | 20 | 10~100 | 保留 top-k token，越小越集中 |
| `top_p` | 0.85 | 0.5~1.0 | 核采样阈值 |
| `repetition_penalty` | 1.1 | 1.0~1.5 | >1.0 抑制重复 |

> 不传任何采样参数时，服务端使用上述保守默认值。

## 模型切换

修改 `config.yaml` 中的 `model.active`：

```yaml
model:
  active: custom_voice  # 改为 base 或 kokoro
```

重启服务即可：
```bash
docker compose restart
```

## 目录结构

```
tts-api/
├── server.py              # 主入口，动态加载 handler
├── config.yaml            # 配置文件
├── tts_client.py          # Python 客户端 SDK
├── handlers/
│   ├── handler_custom_voice.py  # CustomVoice 模型
│   ├── handler_base.py          # Base 模型（Voice Clone）
│   └── handler_kokoro.py        # Kokoro-82M
├── tests/
│   ├── test_api.py        # API 接口测试（unittest）
│   ├── test_tts.py        # 广播剧生成脚本
│   └── test_download_base.py    # Base 模型下载验证
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## 测试

```bash
# 启动服务后运行
cd tests
python -m pytest test_api.py -v

# 或直接运行
python test_api.py
```
