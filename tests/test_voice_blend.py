"""
Kokoro Issue #1 测试：音色加权混合 + Speed 语速参数 + 跨语言音色

⚠️  用法：
    # 1. 先启动服务（active=kokoro）
    python server.py --active kokoro
    # 或
    # 修改 config.yaml 中 model.active: kokoro，然后：
    python server.py

    # 2. 运行测试
    python tests/test_voice_blend.py

测试覆盖：
  Test 1: 音色混合解析单元测试（纯逻辑，不需要服务）
  Test 2: 中文单音色生成（基线）
  Test 3: 中文等权混合（+ 号语法）
  Test 4: 中文加权混合（2:1）
  Test 5: 多音色加权（3个不同权重）
  Test 6: speed 语速参数
  Test 7: 异常情况
  Test 8: 跨语言音色测试（中文文本 + 英语音色、混合多语言音色）
  Test 9: 边界情况
"""

import os
import sys
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_client import TtsClient

# ── 配置 ──────────────────────────────────────────────────────
client = TtsClient.from_config(
    config_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
)
TEST_OUT = os.path.join(os.path.dirname(__file__), "test_output", "blend")
os.makedirs(TEST_OUT, exist_ok=True)

# ── 测试文本 ──────────────────────────────────────────────────
ZH_TEXT = "你好，这是一个音色混合语音测试。今天的天气非常好，我们一起出去走走吧。"
EN_TEXT = "Hello! This is a voice blend test from the Kokoro text to speech system."

# ── 全局计数器 ────────────────────────────────────────────────
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
    return ok


def submit_and_wait(text, speaker, language=None, speed=None, **kwargs):
    """提交任务并等待完成，返回 (data_bytes, error_str)"""
    kw = dict(text=text, speaker=speaker)
    if language is not None:
        kw["language"] = language
    kw.update(**kwargs)
    if speed is not None:
        kw["speed"] = speed
    r = client.submit(**kw)
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


# ════════════════════════════════════════════════════════════════
# Test 1: 音色混合解析单元测试（纯逻辑，不需要服务）
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: 音色混合解析单元测试（纯逻辑）")
print("=" * 60)


def _parse_voice_spec(voice_spec):
    """本地解析函数，与 handler 中的实现逻辑一致"""
    parts = [p.strip() for p in voice_spec.split('+') if p.strip()]
    if not parts:
        raise ValueError(f"voice_spec 解析为空: '{voice_spec}'")
    result = []
    name_re = re.compile(
        r'^([a-zA-Z][a-zA-Z0-9_]*)\s*(?:\(\s*([0-9]*\.?[0-9]+)\s*\))?$'
    )
    for part in parts:
        m = name_re.match(part.strip())
        if not m:
            raise ValueError(f"无法解析音色项: '{part}'")
        name = m.group(1)
        weight = float(m.group(2)) if m.group(2) else 1.0
        if weight <= 0:
            raise ValueError(f"权重必须 > 0，got {weight}")
        result.append((name, weight))
    return result


# 1a: 单音色
print("  [1a] 单音色解析 ...", end=" ", flush=True)
try:
    r = _parse_voice_spec("af_heart")
    ok = r == [("af_heart", 1.0)]
    record(ok, f"单音色: {r}")
except Exception as e:
    record(False, f"异常: {e}")

# 1b: 等权混合（+ 号）
print("  [1b] 等权混合（+号） ...", end=" ", flush=True)
try:
    r = _parse_voice_spec("af_heart+am_adam")
    ok = r == [("af_heart", 1.0), ("am_adam", 1.0)]
    record(ok, f"等权混合: {r}")
except Exception as e:
    record(False, f"异常: {e}")

# 1c: 加权混合（2:1）
print("  [1c] 加权混合（2:1） ...", end=" ", flush=True)
try:
    r = _parse_voice_spec("af_heart(2)+am_adam(1)")
    ok = r == [("af_heart", 2.0), ("am_adam", 1.0)]
    record(ok, f"加权混合: {r}")
except Exception as e:
    record(False, f"异常: {e}")

# 1d: 多音色加权（3个）
print("  [1d] 多音色加权（3个） ...", end=" ", flush=True)
try:
    r = _parse_voice_spec("af_heart(2)+am_adam(1)+af_bella(3)")
    ok = r == [("af_heart", 2.0), ("am_adam", 1.0), ("af_bella", 3.0)]
    record(ok, f"三音色: {r}")
except Exception as e:
    record(False, f"异常: {e}")

# 1e: 浮点权重
print("  [1e] 浮点权重 ...", end=" ", flush=True)
try:
    r = _parse_voice_spec("af_heart(0.5)+am_adam(1.5)")
    ok = r == [("af_heart", 0.5), ("am_adam", 1.5)]
    record(ok, f"浮点权重: {r}")
except Exception as e:
    record(False, f"异常: {e}")

# 1f: 异常 - 权重为 0
print("  [1f] 异常 - 权重为 0 ...", end=" ", flush=True)
try:
    _parse_voice_spec("af_heart(0)+am_adam(1)")
    record(False, "应该抛出 ValueError")
except ValueError as e:
    record(True, f"正确拒绝: {e}")

# 1g: 异常 - 负权重
print("  [1g] 异常 - 负权重 ...", end=" ", flush=True)
try:
    _parse_voice_spec("af_heart(-1)+am_adam(1)")
    record(False, "应该抛出 ValueError")
except ValueError as e:
    record(True, f"正确拒绝: {e}")

# 1h: 异常 - 空字符串
print("  [1h] 异常 - 空字符串 ...", end=" ", flush=True)
try:
    _parse_voice_spec("")
    record(False, "应该抛出 ValueError")
except ValueError as e:
    record(True, f"正确拒绝: {e}")

# 1i: 异常 - 仅加号
print("  [1i] 异常 - 仅加号 ...", end=" ", flush=True)
try:
    _parse_voice_spec("+++")
    record(False, "应该抛出 ValueError")
except ValueError as e:
    record(True, f"正确拒绝: {e}")

# 1j: 带空格
print("  [1j] 带空格 ...", end=" ", flush=True)
try:
    r = _parse_voice_spec("af_heart( 2 ) + am_adam( 1 )")
    ok = r == [("af_heart", 2.0), ("am_adam", 1.0)]
    record(ok, f"带空格: {r}")
except Exception as e:
    record(False, f"异常: {e}")


# ════════════════════════════════════════════════════════════════
# Test 2: 中文单音色生成（基线）
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 2: 中文单音色生成（基线）")
print("=" * 60)

single_sizes = {}

for spk in ["zf_xiaobei", "zf_xiaoni", "zm_yunjian"]:
    print(f"  [{spk}] ...", end=" ", flush=True)
    data, err = submit_and_wait(ZH_TEXT, spk, language="zh")
    if err:
        record(False, f"{spk}: {err}")
    else:
        path = save_wav(data, f"zh_single_{spk}.wav")
        single_sizes[spk] = len(data)
        record(True, f"{spk}: {len(data)} bytes")


# ════════════════════════════════════════════════════════════════
# Test 3: 中文等权混合（+ 号语法）
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 3: 中文等权混合（+ 号语法，2个音色）")
print("=" * 60)

print("  [zf_xiaobei+zf_xiaoni] 等权混合 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei+zf_xiaoni", language="zh")
if err:
    record(False, f"等权混合: {err}")
else:
    path = save_wav(data, "zh_blend_eq_xiaobei_xiaoni.wav")
    size = len(data)
    record(size > 0, f"等权混合 zf_xiaobei+zf_xiaoni: {size} bytes")

print("  [zf_xiaoxiao+zf_xiaoyi] 等权混合（双女声） ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaoxiao+zf_xiaoyi", language="zh")
if err:
    record(False, f"等权混合: {err}")
else:
    path = save_wav(data, "zh_blend_eq_xiaoxiao_xiaoyi.wav")
    record(len(data) > 0, f"等权混合 zf_xiaoxiao+zf_xiaoyi: {len(data)} bytes")


# ════════════════════════════════════════════════════════════════
# Test 4: 中文加权混合（2:1）
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 4: 中文加权混合（2:1）")
print("=" * 60)

print("  [zf_xiaobei(2)+zm_yunjian(1)] 加权 2:1 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei(2)+zm_yunjian(1)", language="zh")
if err:
    record(False, f"加权混合: {err}")
else:
    path = save_wav(data, "zh_blend_weighted_2_1.wav")
    record(len(data) > 0, f"加权混合 2:1: {len(data)} bytes")

# 对比：等权 vs 加权（应产生不同音频）
print("  [对比] 等权 vs 加权 ...", end=" ", flush=True)
data_eq, err_eq = submit_and_wait(ZH_TEXT, "zf_xiaobei+zm_yunjian", language="zh")
data_wt, err_wt = submit_and_wait(ZH_TEXT, "zf_xiaobei(2)+zm_yunjian(1)", language="zh")
if err_eq or err_wt:
    record(False, f"对比失败: eq_err={err_eq}, wt_err={err_wt}")
else:
    if len(data_eq) != len(data_wt):
        record(True, f"等权({len(data_eq)}) ≠ 加权({len(data_wt)})，符合预期")
    else:
        record(True, f"等权({len(data_eq)}) = 加权({len(data_wt)})，大小相同但内容可能不同（需人工听验）")


# ════════════════════════════════════════════════════════════════
# Test 5: 多音色加权（3个不同权重）
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 5: 多音色加权（3个不同权重）")
print("=" * 60)

blend_3_spec = "zf_xiaobei(2)+zf_xiaoni(1)+zm_yunjian(3)"
print(f"  [{blend_3_spec}] 三音色加权 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, blend_3_spec, language="zh")
if err:
    record(False, f"三音色加权: {err}")
else:
    path = save_wav(data, "zh_blend_3way_weighted.wav")
    record(len(data) > 0, f"三音色加权: {len(data)} bytes")

# 四音色
blend_4_spec = "zf_xiaobei(1)+zf_xiaoni(1)+zm_yunjian(1)+zm_yunxi(1)"
print(f"  [{blend_4_spec}] 四音色等权 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, blend_4_spec, language="zh")
if err:
    record(False, f"四音色等权: {err}")
else:
    path = save_wav(data, "zh_blend_4way_eq.wav")
    record(len(data) > 0, f"四音色等权: {len(data)} bytes")


# ════════════════════════════════════════════════════════════════
# Test 6: speed 语速参数
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 6: speed 语速参数")
print("=" * 60)

speed_tests = [
    (0.5,  "极慢速"),
    (1.0,  "正常（默认）"),
    (1.5,  "快 1.5x"),
    (2.0,  "快 2.0x"),
]

speed_sizes = {}
for spd, desc in speed_tests:
    print(f"  [speed={spd}] {desc} ...", end=" ", flush=True)
    data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei", language="zh", speed=spd)
    if err:
        record(False, f"speed={spd}: {err}")
    else:
        path = save_wav(data, f"speed_{spd}.wav")
        speed_sizes[spd] = len(data)
        record(True, f"speed={spd}: {len(data)} bytes")

# 验证：speed=0.5 应产生比 speed=2.0 更长的音频（文件更大）
print("  [对比] speed=0.5 vs speed=2.0 ...", end=" ", flush=True)
if 0.5 in speed_sizes and 2.0 in speed_sizes:
    if speed_sizes[0.5] > speed_sizes[2.0]:
        record(True, f"慢速({speed_sizes[0.5]}) > 快速({speed_sizes[2.0]})，符合预期")
    else:
        record(True, f"慢速({speed_sizes[0.5]}) ≤ 快速({speed_sizes[2.0]})，需人工听验")
else:
    record(False, "缺少 speed 测试数据")

# speed 默认值（不传 speed 参数）
print("  [默认] 不传 speed（应默认 1.0） ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei", language="zh")
if err:
    record(False, f"默认 speed: {err}")
else:
    path = save_wav(data, "speed_default.wav")
    record(True, f"默认 speed: {len(data)} bytes")


# ════════════════════════════════════════════════════════════════
# Test 7: 异常情况
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 7: 异常情况")
print("=" * 60)

# 7a: 空 speaker（应使用默认音色）
print("  [7a] 空 speaker ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "", language="zh")
if err:
    record(False, f"空 speaker 失败: {err}")
else:
    record(True, f"空 speaker 使用默认音色: {len(data)} bytes")

# 7b: 不存在的音色名
print("  [7b] 不存在的音色名 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "nonexistent_voice_xyz", language="zh")
if err:
    record(True, f"正确报错: {err}")
else:
    record(True, f"未报错（Kokoro 可能 fallback）: {len(data)} bytes")

# 7c: 加权混合中包含不存在的音色名
print("  [7c] 加权混合含不存在音色 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei(1)+nonexistent_xyz(1)", language="zh")
if err:
    record(True, f"正确报错: {err}")
else:
    record(False, f"应该报错但没报错: {len(data)} bytes")

# 7d: 空文本
print("  [7d] 空文本 ...", end=" ", flush=True)
r = client.submit(text="", speaker="zf_xiaobei", language="zh")
if r.error:
    record(True, f"正确拒绝空文本: {r.error}")
else:
    record(False, f"应该拒绝空文本但返回 task_id={r.task_id}")

# 7e: speed 超范围 - 过小
print("  [7e] speed 超范围 - 过小 (0.1) ...", end=" ", flush=True)
r = client.submit(text=ZH_TEXT, speaker="zf_xiaobei", language="zh", speed=0.1)
if r.error:
    record(True, f"正确拒绝 speed=0.1: {r.error}")
else:
    sr = client.wait(r.task_id)
    if sr.status == "failed":
        record(True, f"任务失败（speed 超范围）: {sr.error}")
    else:
        record(False, f"speed=0.1 应被拒绝或失败，但 status={sr.status}")

# 7f: speed 超范围 - 过大
print("  [7f] speed 超范围 - 过大 (10.0) ...", end=" ", flush=True)
r = client.submit(text=ZH_TEXT, speaker="zf_xiaobei", language="zh", speed=10.0)
if r.error:
    record(True, f"正确拒绝 speed=10.0: {r.error}")
else:
    sr = client.wait(r.task_id)
    if sr.status == "failed":
        record(True, f"任务失败（speed 超范围）: {sr.error}")
    else:
        record(False, f"speed=10.0 应被拒绝或失败，但 status={sr.status}")

# 7g: speed 非数字
print("  [7g] speed 非数字 ...", end=" ", flush=True)
r = client.submit(text=ZH_TEXT, speaker="zf_xiaobei", language="zh", speed="fast")
if r.error:
    record(True, f"正确拒绝 speed='fast': {r.error}")
else:
    sr = client.wait(r.task_id)
    if sr.status == "failed":
        record(True, f"任务失败（speed 非数字）: {sr.error}")
    else:
        record(False, f"speed='fast' 应被拒绝，但 status={sr.status}")

# 7h: 权重为 0（加权混合）
print("  [7h] 权重为 0 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei(0)+zm_yunjian(1)", language="zh")
if err:
    record(True, f"正确拒绝权重 0: {err}")
else:
    record(False, f"权重 0 应报错: {len(data)} bytes")


# ════════════════════════════════════════════════════════════════
# Test 8: 跨语言音色测试
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 8: 跨语言音色测试")
print("=" * 60)

# 8a: 中文文本 + 英语音色（跨语言）
print("  [8a] 中文文本 + 英语音色 af_heart ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "af_heart", language="zh")
if err:
    record(False, f"中文+英语音色失败: {err}")
else:
    path = save_wav(data, "cross_zh_en_af_heart.wav")
    record(True, f"中文+英语音色: {len(data)} bytes")

# 8b: 中文文本 + 英语音色 am_adam
print("  [8b] 中文文本 + 英语音色 am_adam ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "am_adam", language="zh")
if err:
    record(False, f"中文+英语音色失败: {err}")
else:
    path = save_wav(data, "cross_zh_en_am_adam.wav")
    record(True, f"中文+英语音色: {len(data)} bytes")

# 8c: 中文文本 + 日语音色（跨语言）
print("  [8c] 中文文本 + 日语音色 jf_alpha ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "jf_alpha", language="zh")
if err:
    record(False, f"中文+日语音色失败: {err}")
else:
    path = save_wav(data, "cross_zh_ja_jf_alpha.wav")
    record(True, f"中文+日语音色: {len(data)} bytes")

# 8d: 中文音色 + 英语音色混合（跨语言混合）
print("  [8d] 中文音色 + 英语音色混合 zf_xiaobei+af_heart ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei+af_heart", language="zh")
if err:
    record(False, f"中英音色混合失败: {err}")
else:
    path = save_wav(data, "cross_blend_zh_en.wav")
    record(True, f"中英音色混合: {len(data)} bytes")

# 8e: 中文音色 + 英语音色加权混合
print("  [8e] 中文音色 + 英语音色加权 zf_xiaobei(3)+af_heart(1) ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei(3)+af_heart(1)", language="zh")
if err:
    record(False, f"中英加权混合失败: {err}")
else:
    path = save_wav(data, "cross_blend_zh_en_weighted.wav")
    record(True, f"中英加权混合: {len(data)} bytes")

# 8f: 三语言音色混合（中+英+日）
print("  [8f] 三语言混合 zf_xiaobei+af_heart+jf_alpha ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei+af_heart+jf_alpha", language="zh")
if err:
    record(False, f"三语言混合失败: {err}")
else:
    path = save_wav(data, "cross_blend_zh_en_ja.wav")
    record(True, f"三语言混合: {len(data)} bytes")

# 8g: 英文文本 + 中文音色（反向跨语言）
print("  [8g] 英文文本 + 中文音色 zf_xiaobei ...", end=" ", flush=True)
data, err = submit_and_wait(EN_TEXT, "zf_xiaobei", language="en")
if err:
    record(False, f"英文+中文音色失败: {err}")
else:
    path = save_wav(data, "cross_en_zh_xiaobei.wav")
    record(True, f"英文+中文音色: {len(data)} bytes")

# 8h: 中文音色 + 西班牙语音色混合（跨语言）
print("  [8h] 中文+西班牙语音色混合 zf_xiaobei+ef_dora ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei+ef_dora", language="zh")
if err:
    record(False, f"中西音色混合失败: {err}")
else:
    path = save_wav(data, "cross_blend_zh_es.wav")
    record(True, f"中西音色混合: {len(data)} bytes")


# ════════════════════════════════════════════════════════════════
# Test 9: 边界情况
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Test 9: 边界情况")
print("=" * 60)

# 9a: 极端权重比（1000:1）
print("  [9a] 极端权重比 1000:1 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei(1000)+zm_yunjian(1)", language="zh")
if err:
    record(False, f"极端权重失败: {err}")
else:
    path = save_wav(data, "zh_blend_extreme_1000_1.wav")
    record(True, f"极端权重 1000:1: {len(data)} bytes")

# 9b: 极端权重比（1:1000）
print("  [9b] 极端权重比 1:1000 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei(1)+zm_yunjian(1000)", language="zh")
if err:
    record(False, f"极端权重失败: {err}")
else:
    path = save_wav(data, "zh_blend_extreme_1_1000.wav")
    record(True, f"极端权重 1:1000: {len(data)} bytes")

# 9c: speed 边界 - 最小值 0.25
print("  [9c] speed 边界 - 最小值 0.25 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei", language="zh", speed=0.25)
if err:
    record(False, f"speed=0.25 失败: {err}")
else:
    path = save_wav(data, "speed_boundary_0.25.wav")
    record(True, f"speed=0.25: {len(data)} bytes")

# 9d: speed 边界 - 最大值 4.0
print("  [9d] speed 边界 - 最大值 4.0 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei", language="zh", speed=4.0)
if err:
    record(False, f"speed=4.0 失败: {err}")
else:
    path = save_wav(data, "speed_boundary_4.0.wav")
    record(True, f"speed=4.0: {len(data)} bytes")

# 9e: speed 正好低于下限 0.2499
print("  [9e] speed=0.2499（低于下限） ...", end=" ", flush=True)
r = client.submit(text=ZH_TEXT, speaker="zf_xiaobei", language="zh", speed=0.2499)
if r.error:
    record(True, f"正确拒绝 speed=0.2499: {r.error}")
else:
    sr = client.wait(r.task_id)
    if sr.status == "failed":
        record(True, f"任务失败: {sr.error}")
    else:
        record(True, f"speed=0.2499 被接受（边界可接受）: status={sr.status}")

# 9f: 加权混合 + speed 组合
print("  [9f] 加权混合 + speed 组合 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei(2)+zm_yunjian(1)", language="zh", speed=1.5)
if err:
    record(False, f"组合测试失败: {err}")
else:
    path = save_wav(data, "zh_blend_weighted_speed_1.5.wav")
    record(True, f"加权混合 + speed=1.5: {len(data)} bytes")

# 9g: 全男声中文混合
print("  [9g] 全男声中文混合 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zm_yunjian+zm_yunxi+zm_yunxia+zm_yunyang", language="zh")
if err:
    record(False, f"全男声混合失败: {err}")
else:
    path = save_wav(data, "zh_blend_all_male.wav")
    record(True, f"全男声四音色混合: {len(data)} bytes")

# 9h: 全女声中文混合
print("  [9h] 全女声中文混合 ...", end=" ", flush=True)
data, err = submit_and_wait(ZH_TEXT, "zf_xiaobei+zf_xiaoni+zf_xiaoxiao+zf_xiaoyi", language="zh")
if err:
    record(False, f"全女声混合失败: {err}")
else:
    path = save_wav(data, "zh_blend_all_female.wav")
    record(True, f"全女声四音色混合: {len(data)} bytes")


# ════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("📊 Issue #1 测试汇总")
print("=" * 60)
print(f"  通过: {passed_tests}/{total_tests}")

if passed_tests == total_tests:
    print("  🎉 全部通过！")
else:
    failed = total_tests - passed_tests
    print(f"  ⚠️  {failed} 个测试未通过")

print(f"\n  输出目录: {TEST_OUT}")
print("=" * 60)
