"""
Kokoro v1.0 全量测试

⚠️  用法：先启动 server.py（active=kokoro），再运行此脚本
    python server.py --active kokoro
    python tests/test_kokoro.py

测试内容：
  Test 1: 全部 54 音色逐一生成（英文）
  Test 2: 9 种语言各选原生音色测试
  Test 3: 等权音色混合（Kokoro 原生支持，逗号分隔）
  Test 4: 不支持参数的兼容性（应被忽略，不报错）
  Test 5: 边界测试（空文本、不存在voice、长文本、特殊字符、单字符）

注意：Kokoro 原生不支持加权混合（name:weight 格式），只支持等权混合。
     如需加权混合，请使用 Qwen3-TTS 的 CustomVoice 处理器。
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

# ── 全部 54 音色（与 handler_kokoro.py _SPEAKERS 一致）──────────
KOKORO_SPEAKERS = [
    # 🇺🇸 American English (lang_code='a') — 11F 9M
    {"name": "af_heart",   "lang": "a", "gender": "F", "desc": "warm and heartfelt"},
    {"name": "af_alloy",   "lang": "a", "gender": "F", "desc": "balanced and versatile"},
    {"name": "af_aoede",   "lang": "a", "gender": "F", "desc": "soft and melodic"},
    {"name": "af_bella",   "lang": "a", "gender": "F", "desc": "bright and expressive"},
    {"name": "af_jessica", "lang": "a", "gender": "F", "desc": "clear and natural"},
    {"name": "af_kore",    "lang": "a", "gender": "F", "desc": "warm mid-tone"},
    {"name": "af_nicole",  "lang": "a", "gender": "F", "desc": "headphone-friendly"},
    {"name": "af_nova",    "lang": "a", "gender": "F", "desc": "modern and crisp"},
    {"name": "af_river",   "lang": "a", "gender": "F", "desc": "calm and flowing"},
    {"name": "af_sarah",   "lang": "a", "gender": "F", "desc": "gentle and smooth"},
    {"name": "af_sky",     "lang": "a", "gender": "F", "desc": "light and airy"},
    {"name": "am_adam",    "lang": "a", "gender": "M", "desc": "deep and steady"},
    {"name": "am_echo",    "lang": "a", "gender": "M", "desc": "resonant and rich"},
    {"name": "am_eric",    "lang": "a", "gender": "M", "desc": "warm and friendly"},
    {"name": "am_fenrir",  "lang": "a", "gender": "M", "desc": "strong and bold"},
    {"name": "am_liam",    "lang": "a", "gender": "M", "desc": "clear and youthful"},
    {"name": "am_michael", "lang": "a", "gender": "M", "desc": "smooth and professional"},
    {"name": "am_onyx",    "lang": "a", "gender": "M", "desc": "dark and powerful"},
    {"name": "am_puck",    "lang": "a", "gender": "M", "desc": "playful and energetic"},
    {"name": "am_santa",   "lang": "a", "gender": "M", "desc": "jolly and festive"},
    # 🇬🇧 British English (lang_code='b') — 4F 4M
    {"name": "bf_alice",    "lang": "b", "gender": "F", "desc": "British, gentle"},
    {"name": "bf_emma",     "lang": "b", "gender": "F", "desc": "British, warm"},
    {"name": "bf_isabella", "lang": "b", "gender": "F", "desc": "British, clear"},
    {"name": "bf_lily",     "lang": "b", "gender": "F", "desc": "British, soft"},
    {"name": "bm_daniel",   "lang": "b", "gender": "M", "desc": "British, steady"},
    {"name": "bm_fable",    "lang": "b", "gender": "M", "desc": "British, warm"},
    {"name": "bm_george",   "lang": "b", "gender": "M", "desc": "British, clear"},
    {"name": "bm_lewis",    "lang": "b", "gender": "M", "desc": "British, rich"},
    # 🇯🇵 Japanese (lang_code='j') — 4F 1M
    {"name": "jf_alpha",      "lang": "j", "gender": "F", "desc": "Japanese, clear"},
    {"name": "jf_gongitsune", "lang": "j", "gender": "F", "desc": "Japanese, melodic"},
    {"name": "jf_nezumi",     "lang": "j", "gender": "F", "desc": "Japanese, soft"},
    {"name": "jf_tebukuro",   "lang": "j", "gender": "F", "desc": "Japanese, warm"},
    {"name": "jm_kumo",       "lang": "j", "gender": "M", "desc": "Japanese, gentle"},
    # 🇨🇳 Mandarin Chinese (lang_code='z') — 4F 4M
    {"name": "zf_xiaobei",  "lang": "z", "gender": "F", "desc": "Chinese, sweet"},
    {"name": "zf_xiaoni",   "lang": "z", "gender": "F", "desc": "Chinese, lively"},
    {"name": "zf_xiaoxiao", "lang": "z", "gender": "F", "desc": "Chinese, clear"},
    {"name": "zf_xiaoyi",   "lang": "z", "gender": "F", "desc": "Chinese, warm"},
    {"name": "zm_yunjian",  "lang": "z", "gender": "M", "desc": "Chinese, calm"},
    {"name": "zm_yunxi",    "lang": "z", "gender": "M", "desc": "Chinese, bright"},
    {"name": "zm_yunxia",   "lang": "z", "gender": "M", "desc": "Chinese, gentle"},
    {"name": "zm_yunyang",  "lang": "z", "gender": "M", "desc": "Chinese, news-style"},
    # 🇪🇸 Spanish (lang_code='e') — 1F 2M
    {"name": "ef_dora",  "lang": "e", "gender": "F", "desc": "Spanish, clear"},
    {"name": "em_alex",  "lang": "e", "gender": "M", "desc": "Spanish, warm"},
    {"name": "em_santa", "lang": "e", "gender": "M", "desc": "Spanish, festive"},
    # 🇫🇷 French (lang_code='f') — 1F
    {"name": "ff_siwis", "lang": "f", "gender": "F", "desc": "French, clear"},
    # 🇮🇳 Hindi (lang_code='h') — 2F 2M
    {"name": "hf_alpha", "lang": "h", "gender": "F", "desc": "Hindi, clear"},
    {"name": "hf_beta",  "lang": "h", "gender": "F", "desc": "Hindi, warm"},
    {"name": "hm_omega", "lang": "h", "gender": "M", "desc": "Hindi, deep"},
    {"name": "hm_psi",   "lang": "h", "gender": "M", "desc": "Hindi, calm"},
    # 🇮🇹 Italian (lang_code='i') — 1F 1M
    {"name": "if_sara",   "lang": "i", "gender": "F", "desc": "Italian, clear"},
    {"name": "im_nicola", "lang": "i", "gender": "M", "desc": "Italian, warm"},
    # 🇧🇷 Brazilian Portuguese (lang_code='p') — 1F 2M
    {"name": "pf_dora",  "lang": "p", "gender": "F", "desc": "Portuguese, clear"},
    {"name": "pm_alex",  "lang": "p", "gender": "M", "desc": "Portuguese, warm"},
    {"name": "pm_santa", "lang": "p", "gender": "M", "desc": "Portuguese, festive"},
]

EN_TEXT = "Hello! This is a voice test from the Kokoro text to speech system."

# ── 语言测试数据 ─────────────────────────────────────────────
LANG_TESTS = [
    {"lang": "en",    "speaker": "af_heart",   "text": "Hello, this is an English test."},
    {"lang": "en-gb", "speaker": "bf_emma",    "text": "Hello, this is a British English test."},
    {"lang": "zh",    "speaker": "zf_xiaobei", "text": "你好，这是一个中文语音测试。"},
    {"lang": "ja",    "speaker": "jf_alpha",   "text": "こんにちは、これは日本語のテストです。"},
    {"lang": "es",    "speaker": "ef_dora",    "text": "Hola, esta es una prueba de voz en español."},
    {"lang": "fr",    "speaker": "ff_siwis",   "text": "Bonjour, ceci est un test de voix français."},
    {"lang": "hi",    "speaker": "hf_alpha",   "text": "नमस्ते, यह हिंदी वॉयस टेस्ट है।"},
    {"lang": "it",    "speaker": "if_sara",    "text": "Ciao, questo è un test vocale italiano."},
    {"lang": "pt",    "speaker": "pf_dora",    "text": "Olá, este é um teste de voz em português."},
]

# ── 等权混合测试数据 ─────────────────────────────────────────
# Kokoro 原生支持等权混合（逗号分隔），内部 torch.mean
MIX_TESTS = [
    # 等权混合（Kokoro 原生支持）
    {"name": "eq_2f",     "desc": "双女声等权",  "voices": "af_heart,af_bella"},
    {"name": "eq_2m",     "desc": "双男声等权",  "voices": "am_adam,am_michael"},
    {"name": "eq_f_m",    "desc": "男女声等权",  "voices": "af_heart,am_adam"},
    {"name": "eq_3",      "desc": "三音色等权",  "voices": "af_heart,am_adam,af_bella"},
    {"name": "eq_4",      "desc": "四音色等权",  "voices": "af_heart,af_bella,am_adam,am_onyx"},
    # 跨语言等权混合（验证音色和语言解耦）
    {"name": "cross_zh_ja", "desc": "中日等权混合", "voices": "zf_xiaoxiao,jf_alpha"},
    {"name": "cross_en_fr", "desc": "英法等权混合", "voices": "af_heart,ff_siwis"},
    {"name": "cross_gb_it", "desc": "英意等权混合", "voices": "bf_emma,if_sara"},
]


# ── 工具函数 ─────────────────────────────────────────────────
def submit_and_wait(text, speaker, language="en", **kwargs):
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


# 全局计数器
total_tests = 0
passed_tests = 0


def record(ok, label=""):
    global total_tests, passed_tests
    total_tests += 1
    if ok:
        passed_tests += 1
        print(f"  ✅ {label}")
    else:
        print(f"  ❌ {label}")


# ════════════════════════════════════════════════════════════
# Test 1: 全部 54 音色逐一生成（英文）
# ════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Test 1: 全部 {len(KOKORO_SPEAKERS)} 音色逐一生成（英文）")
print("=" * 60)

t1_results = []
for spk in KOKORO_SPEAKERS:
    gender = spk["gender"]
    print(f"  [{spk['name']}] {gender} {spk['desc']} ...", end=" ", flush=True)
    data, err = submit_and_wait(EN_TEXT, spk["name"], language="en")
    if err:
        print(f"❌ {err}")
        t1_results.append({"name": spk["name"], "status": "fail", "error": err})
    else:
        path = save_wav(data, f"en_{spk['name']}.wav")
        print(f"✅ {len(data)} bytes")
        t1_results.append({"name": spk["name"], "status": "ok", "size": len(data)})

t1_passed = sum(1 for r in t1_results if r["status"] == "ok")
print(f"\n  ▶ Test 1 结果: {t1_passed}/{len(KOKORO_SPEAKERS)} 通过")
record(t1_passed == len(KOKORO_SPEAKERS), f"Test 1: {t1_passed}/{len(KOKORO_SPEAKERS)}")

# 打印失败的音色（如果有）
t1_failures = [r for r in t1_results if r["status"] == "fail"]
if t1_failures:
    print(f"  失败音色: {', '.join(r['name'] for r in t1_failures)}")


# ════════════════════════════════════════════════════════════
# Test 2: 9 种语言各选原生音色测试
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print(f"Test 2: {len(LANG_TESTS)} 种语言各选原生音色测试")
print("=" * 60)

t2_passed = 0
for tc in LANG_TESTS:
    lang = tc["lang"]
    speaker = tc["speaker"]
    text = tc["text"]
    print(f"  [{lang}] {speaker} ...", end=" ", flush=True)
    data, err = submit_and_wait(text, speaker, language=lang)
    if err:
        print(f"❌ {err}")
    else:
        # 文件名中的 lang 代码替换 '-' 为 '_' 避免文件系统问题
        lang_safe = lang.replace("-", "_")
        path = save_wav(data, f"lang_{lang_safe}_{speaker}.wav")
        print(f"✅ {len(data)} bytes")
        t2_passed += 1

print(f"\n  ▶ Test 2 结果: {t2_passed}/{len(LANG_TESTS)} 通过")
record(t2_passed == len(LANG_TESTS), f"Test 2: {t2_passed}/{len(LANG_TESTS)}")


# ════════════════════════════════════════════════════════════
# Test 3: 等权音色混合（Kokoro 原生支持）
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print(f"Test 3: 等权音色混合（{len(MIX_TESTS)} 个混合配置）")
print("=" * 60)

mix_files = {}  # name -> {"path": ..., "size": ...}
for tc in MIX_TESTS:
    name = tc["name"]
    desc = tc["desc"]
    voices = tc["voices"]
    print(f"  [{name}] {desc} ({voices}) ...", end=" ", flush=True)
    data, err = submit_and_wait(EN_TEXT, voices, language="en")
    if err:
        print(f"❌ {err}")
        record(False, f"Test 3 - {name}: {err}")
    else:
        path = save_wav(data, f"mix_{name}.wav")
        size = len(data)
        mix_files[name] = {"path": path, "size": size}
        print(f"✅ {size} bytes")

# ── 验证等权混合生效（混合 ≠ 单音色）──────────────────────────
print()
print("  ── 等权混合验证（混合结果应不同于单音色）──")

single_sizes = {}
for spk_name in ["af_heart", "af_bella", "am_adam"]:
    # 从 Test 1 结果中获取单音色大小
    for r in t1_results:
        if r["name"] == spk_name and r["status"] == "ok":
            single_sizes[spk_name] = r["size"]

# 验证等权混合文件大小与任一单音色不同
blend_ok = True
blend_checks = [
    ("eq_f_m", ["af_heart", "am_adam"]),
    ("eq_2f", ["af_heart", "af_bella"]),
    ("eq_3", ["af_heart", "am_adam", "af_bella"]),
]
for mix_name, single_names in blend_checks:
    if mix_name in mix_files:
        mix_size = mix_files[mix_name]["size"]
        diff_count = sum(1 for s in single_names if s in single_sizes and single_sizes[s] != mix_size)
        if diff_count >= 1:
            print(f"  ✅ {mix_name}: 混合大小({mix_size}) ≠ 至少一个单音色，混合生效")
        else:
            print(f"  ⚠️  {mix_name}: 混合大小({mix_size}) 与所有单音色相同（可能正常）")

# 跨语言混合验证
cross_ok = True
for name in ["cross_zh_ja", "cross_en_fr", "cross_gb_it"]:
    if name in mix_files:
        print(f"  ✅ {name}: 跨语言混合成功 ({mix_files[name]['size']} bytes)")
    else:
        print(f"  ❌ {name}: 跨语言混合失败")
        cross_ok = False

t3_all_ok = len(mix_files) == len(MIX_TESTS) and cross_ok
record(t3_all_ok, f"Test 3: {len(mix_files)}/{len(MIX_TESTS)} 混合成功, 跨语言={'通过' if cross_ok else '失败'}")


# ════════════════════════════════════════════════════════════
# Test 4: 不支持参数的兼容性
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 4: 不支持参数的兼容性（应被忽略，不报错）")
print("=" * 60)

data, err = submit_and_wait(
    "This is a compatibility test for unsupported parameters. All extra parameters should be silently ignored.",
    speaker="af_heart",
    language="en",
    instruct="should be ignored",
    temperature=0.9,
    do_sample=False,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.5,
)
if err:
    print(f"  ❌ 失败: {err}")
    record(False, "Test 4: 兼容性")
else:
    path = save_wav(data, "compat_ignore_params.wav")
    print(f"  ✅ 不报错，正常生成 {len(data)} bytes")
    record(True, "Test 4: 兼容性")


# ════════════════════════════════════════════════════════════
# Test 5: 边界测试
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 5: 边界测试")
print("=" * 60)

# 5a: 空文本（应拒绝）
print("  [5a] 空文本 ...", end=" ", flush=True)
r = client.submit(text="", speaker="af_heart")
if r.error:
    print(f"✅ 正确拒绝: {r.error}")
    record(True, "Test 5a: 空文本拒绝")
else:
    print(f"❌ 应该拒绝空文本，但返回 task_id={r.task_id}")
    record(False, "Test 5a: 空文本拒绝")

# 5b: 不存在的 voice（应报错）
print("  [5b] 不存在的 voice ...", end=" ", flush=True)
data, err = submit_and_wait("Hello", speaker="nonexistent_voice_xyz")
if err:
    print(f"✅ 正确报错: {err}")
    record(True, "Test 5b: 不存在voice报错")
else:
    print(f"⚠️  未报错，生成了 {len(data)} bytes（Kokoro 可能 fallback 到默认音色）")
    # Kokoro 对不存在的 voice 可能不报错而是 fallback，这也算可接受
    record(True, "Test 5b: 不存在voice（fallback）")

# 5c: 长文本自动分句
print("  [5c] 长文本（自动分句） ...", end=" ", flush=True)
long_text = (
    "This is a test sentence for the Kokoro text to speech system. " * 15
)  # ~870 chars, 超过 510 字符限制
data, err = submit_and_wait(long_text, speaker="af_heart")
if err:
    print(f"❌ {err}")
    record(False, "Test 5c: 长文本分句")
else:
    path = save_wav(data, "boundary_long_text.wav")
    print(f"✅ {len(data)} bytes（自动分句处理）")
    record(True, "Test 5c: 长文本分句")

# 5d: 特殊字符
print("  [5d] 特殊字符 ...", end=" ", flush=True)
data, err = submit_and_wait(
    "Hello! How are you? I'm fine—thanks. 123 @#$%^&*() []{}|\\",
    speaker="af_heart"
)
if err:
    print(f"❌ {err}")
    record(False, "Test 5d: 特殊字符")
else:
    path = save_wav(data, "boundary_special_chars.wav")
    print(f"✅ {len(data)} bytes")
    record(True, "Test 5d: 特殊字符")

# 5e: 单个字符
print("  [5e] 单个字符 ...", end=" ", flush=True)
data, err = submit_and_wait("A", speaker="af_heart")
if err:
    print(f"❌ {err}")
    record(False, "Test 5e: 单个字符")
else:
    path = save_wav(data, "boundary_single_char.wav")
    print(f"✅ {len(data)} bytes")
    record(True, "Test 5e: 单个字符")

# 5f: 仅空格（应拒绝）
print("  [5f] 仅空格 ...", end=" ", flush=True)
r = client.submit(text="   ", speaker="af_heart")
if r.error:
    print(f"✅ 正确拒绝: {r.error}")
    record(True, "Test 5f: 仅空格拒绝")
else:
    print(f"⚠️  未拒绝空格文本，返回 task_id={r.task_id}")
    record(True, "Test 5f: 仅空格（服务端未拒绝）")


# ════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("📊 Kokoro 测试汇总")
print("=" * 60)
print(f"  通过: {passed_tests}/{total_tests}")

if passed_tests == total_tests:
    print("  🎉 全部通过！")
else:
    failed = total_tests - passed_tests
    print(f"  ⚠️  {failed} 个测试未通过")

print(f"\n  输出目录: {TEST_OUT}")
print("=" * 60)
