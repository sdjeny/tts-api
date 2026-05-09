"""
Kokoro 模型测试

用法：先启动 server.py（active=kokoro），再运行此脚本

测试内容：
  1. 全部 18 音色逐一生成（英文）
  2. 中文音色测试
  3. 多音色混合（Kokoro 原生支持，逗号分隔）
  4. 不支持参数的兼容性
  5. 边界测试
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_client import TtsClient

client = TtsClient.from_config(
    config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
)

TEST_OUT = os.path.join(os.path.dirname(__file__), "test_output", "kokoro")
os.makedirs(TEST_OUT, exist_ok=True)

# ── 全部音色 ───────────────────────────────────────────────
KOKORO_SPEAKERS = [
    {"name": "af_heart",   "desc": "Female, warm and heartfelt"},
    {"name": "af_alloy",   "desc": "Female, balanced and versatile"},
    {"name": "af_aoede",   "desc": "Female, soft and melodic"},
    {"name": "af_bella",   "desc": "Female, bright and expressive"},
    {"name": "af_jessica", "desc": "Female, clear and natural"},
    {"name": "af_kore",    "desc": "Female, warm mid-tone"},
    {"name": "af_nova",    "desc": "Female, modern and crisp"},
    {"name": "af_river",   "desc": "Female, calm and flowing"},
    {"name": "af_sarah",   "desc": "Female, gentle and smooth"},
    {"name": "af_sky",     "desc": "Female, light and airy"},
    {"name": "am_adam",    "desc": "Male, deep and steady"},
    {"name": "am_echo",    "desc": "Male, resonant and rich"},
    {"name": "am_eric",    "desc": "Male, warm and friendly"},
    {"name": "am_fenrir",  "desc": "Male, strong and bold"},
    {"name": "am_liam",    "desc": "Male, clear and youthful"},
    {"name": "am_michael", "desc": "Male, smooth and professional"},
    {"name": "am_onyx",    "desc": "Male, dark and powerful"},
    {"name": "am_puck",    "desc": "Male, playful and energetic"},
    {"name": "am_santa",   "desc": "Male, jolly and festive"},
]

EN_TEXT = "Hello! This is a voice test from the Kokoro text to speech system. How does this voice sound?"
ZH_TEXT = "你好！这是一个语音合成测试。今天天气真不错，我们一起去公园散步吧。"


# ── 工具函数 ───────────────────────────────────────────────
def submit_and_wait(text, speaker, language="English", **kwargs):
    """提交任务并等待完成，返回 (data_bytes, error_str)"""
    r = client.submit(text=text, language=language, speaker=speaker, **kwargs)
    if r.error:
        return None, r.error
    sr = client.wait(r.task_id)
    if not sr.ok:
        return None, f"{sr.status}: {sr.error}"
    dl = client.download(r.task_id)
    if not dl.ok:
        return None, dl.error
    return dl.data, None


def save_wav(data, filename):
    path = os.path.join(TEST_OUT, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ════════════════════════════════════════════════════════════
# Test 1: 全部音色逐一生成（英文）
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: 全部 18 音色逐一生成（英文）")
print("=" * 60)

results = []
for spk in KOKORO_SPEAKERS:
    print(f"  [{spk['name']}] {spk['desc']} ...", end=" ", flush=True)
    data, err = submit_and_wait(EN_TEXT, spk["name"], language="English")
    if err:
        print(f"❌ {err}")
        results.append({"name": spk["name"], "status": "fail", "error": err})
    else:
        path = save_wav(data, f"en_{spk['name']}.wav")
        print(f"✅ {len(data)} bytes")
        results.append({"name": spk["name"], "status": "ok", "size": len(data)})

passed = sum(1 for r in results if r["status"] == "ok")
print(f"\n结果: {passed}/{len(KOKORO_SPEAKERS)} 通过")


# ════════════════════════════════════════════════════════════
# Test 2: 中文音色测试
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 2: 中文音色测试")
print("=" * 60)

zh_speakers = ["af_heart", "af_bella", "am_adam", "am_michael"]
for spk in zh_speakers:
    print(f"  [{spk}] 中文 ...", end=" ", flush=True)
    data, err = submit_and_wait(ZH_TEXT, spk, language="Chinese")
    if err:
        print(f"❌ {err}")
    else:
        path = save_wav(data, f"zh_{spk}.wav")
        print(f"✅ {len(data)} bytes")


# ════════════════════════════════════════════════════════════
# Test 3: 多音色混合（Kokoro 原生支持）
# ════════════════════════════════════════════════════════════
# Kokoro pipeline.load_voice() 原生支持逗号分隔多音色，内部做平均
print()
print("=" * 60)
print("Test 3: 多音色混合（Kokoro 原生支持）")
print("=" * 60)

# 注意：Kokoro 原生混合是等权平均，不支持自定义比例
# 如果需要自定义比例，需要修改 handler 或多次生成后 waveform 叠加
MIX_PROFILES = [
    {"name": "mix_2f",    "desc": "双女声混合",  "voices": "af_heart,af_bella"},
    {"name": "mix_2m",    "desc": "双男声混合",  "voices": "am_adam,am_michael"},
    {"name": "mix_f_m",   "desc": "男女声混合",  "voices": "af_heart,am_adam"},
    {"name": "mix_3",     "desc": "三音色混合",  "voices": "af_heart,am_adam,af_bella"},
    {"name": "mix_4",     "desc": "四音色混合",  "voices": "af_bella,af_sky,am_eric,am_onyx"},
]

for profile in MIX_PROFILES:
    print(f"  [{profile['name']}] {profile['desc']} ({profile['voices']}) ...", end=" ", flush=True)
    # 多音色直接传逗号分隔字符串
    data, err = submit_and_wait(EN_TEXT, profile["voices"], language="English")
    if err:
        print(f"❌ {err}")
    else:
        path = save_wav(data, f"{profile['name']}.wav")
        print(f"✅ {len(data)} bytes")


# ════════════════════════════════════════════════════════════
# Test 4: 不支持参数的兼容性
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 4: 不支持参数的兼容性（应被忽略，不报错）")
print("=" * 60)

data, err = submit_and_wait(
    "This is a compatibility test for unsupported parameters.",
    speaker="af_heart",
    language="English",
    instruct="should be ignored",
    temperature=0.9,
    do_sample=False,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.5,
)
if err:
    print(f"  ❌ 失败: {err}")
else:
    path = save_wav(data, "compat_ignore_params.wav")
    print(f"  ✅ 不报错，正常生成 {len(data)} bytes")


# ════════════════════════════════════════════════════════════
# Test 5: 边界测试
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 5: 边界测试")
print("=" * 60)

# 5a: 空文本
print("  [5a] 空文本 ...", end=" ", flush=True)
r = client.submit(text="", speaker="af_heart")
if r.error:
    print(f"✅ 正确拒绝: {r.error}")
else:
    print(f"❌ 应该拒绝空文本，但返回 task_id={r.task_id}")

# 5b: 不存在的 voice
print("  [5b] 不存在的 voice ...", end=" ", flush=True)
data, err = submit_and_wait("Hello", speaker="nonexistent_voice_xyz")
if err:
    print(f"✅ 正确报错: {err}")
else:
    print(f"⚠️  未报错，生成了 {len(data)} bytes（Kokoro 可能不校验 voice 是否存在）")

# 5c: 超长文本（Kokoro 单段上限 510 字符，但 pipeline 会自动分句）
print("  [5c] 长文本（自动分句） ...", end=" ", flush=True)
long_text = "This is a test sentence. " * 50  # ~1100 chars
data, err = submit_and_wait(long_text, speaker="af_heart")
if err:
    print(f"❌ {err}")
else:
    path = save_wav(data, "boundary_long_text.wav")
    print(f"✅ {len(data)} bytes（自动分句处理）")

# 5d: 特殊字符
print("  [5d] 特殊字符 ...", end=" ", flush=True)
data, err = submit_and_wait("Hello! How are you? I'm fine—thanks. 123 @#$%", speaker="af_heart")
if err:
    print(f"❌ {err}")
else:
    path = save_wav(data, "boundary_special_chars.wav")
    print(f"✅ {len(data)} bytes")

# 5e: 单个字母/数字
print("  [5e] 单个字符 ...", end=" ", flush=True)
data, err = submit_and_wait("A", speaker="af_heart")
if err:
    print(f"❌ {err}")
else:
    path = save_wav(data, "boundary_single_char.wav")
    print(f"✅ {len(data)} bytes")


# ════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Kokoro 测试完成")
print(f"输出目录: {TEST_OUT}")
print("=" * 60)
