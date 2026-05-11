"""
Kokoro 中文音色基线测试
测试全部 8 个中文音色（4女4男），单音色生成，不混合。

用法：
    python tests/test_zh_voices.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_client import TtsClient

client = TtsClient.from_config(
    config_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
)
TEST_OUT = os.path.join(os.path.dirname(__file__), "test_output", "zh_voices")
os.makedirs(TEST_OUT, exist_ok=True)

ZH_TEXT = "你好，这是一个中文音色测试。今天天气真不错，我们一起出去走走吧。"

# ── 全部中文音色 ──────────────────────────────────────────────
VOICES = [
    # 女声
    {"name": "zf_xiaobei",  "gender": "F", "desc": "甜美"},
    {"name": "zf_xiaoni",   "gender": "F", "desc": "活泼"},
    {"name": "zf_xiaoxiao", "gender": "F", "desc": "清晰"},
    {"name": "zf_xiaoyi",   "gender": "F", "desc": "温暖"},
    # 男声
    {"name": "zm_yunjian",  "gender": "M", "desc": "平静"},
    {"name": "zm_yunxi",    "gender": "M", "desc": "明亮"},
    {"name": "zm_yunxia",   "gender": "M", "desc": "温柔"},
    {"name": "zm_yunyang",  "gender": "M", "desc": "新闻"},
]

total = 0
passed = 0
failed = []

print("=" * 60)
print("Kokoro 中文音色基线测试（单音色）")
print("=" * 60)

for v in VOICES:
    name = v["name"]
    label = f"{name} ({v['gender']}, {v['desc']})"
    print(f"  [{label}] ...", end=" ", flush=True)

    total += 1
    start = time.time()

    r = client.submit(text=ZH_TEXT, speaker=name, language="zh")
    if r.error:
        print(f"❌ 提交失败: {r.error}")
        failed.append(label)
        continue

    sr = client.wait(r.task_id)
    elapsed = time.time() - start

    if not sr.ok:
        print(f"❌ {sr.status}: {sr.error} ({elapsed:.1f}s)")
        failed.append(label)
        continue

    dl = client.download(r.task_id)
    if not dl.ok:
        print(f"❌ 下载失败: {dl.error} ({elapsed:.1f}s)")
        failed.append(label)
        continue

    path = os.path.join(TEST_OUT, f"{name}.wav")
    with open(path, "wb") as f:
        f.write(dl.data)

    passed += 1
    print(f"✅ {len(dl.data)} bytes, {elapsed:.1f}s")

# ── 汇总 ──────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"📊 中文音色测试: {passed}/{total}")
if failed:
    print(f"  ❌ 失败: {', '.join(failed)}")
else:
    print("  🎉 全部通过！")
print(f"  输出目录: {TEST_OUT}")
print("=" * 60)
