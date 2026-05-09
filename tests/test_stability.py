"""
Test 7: 双角色音色稳定性测试
==============================
同一段对白，Dylan 和 Vivian 两个角色：
  场景1 ababab：交替对话，验证交叉对话中音色是否稳定
  场景2 aaaabbbb：各自连续对话，验证连续对话中音色是否稳定

判断标准：同一角色多次生成的音频文件大小差异 <30%
"""
import os
import sys
import hashlib
import struct
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_client import TtsClient

client = TtsClient.from_config(
    config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
)

TEST_OUT = os.path.join(os.path.dirname(__file__), "test_output", "stability")
os.makedirs(TEST_OUT, exist_ok=True)

# ── 同一段对白，6轮交替 ──────────────────────────────────────
# 场景1 ababab：A= Dylan, B= Vivian
DIALOGUE = [
    ("A", "Dylan", "今天天气真不错，我们一起去公园散步吧。"),
    ("B", "Vivian", "好啊，我正好也想出去走走。你想去哪里？"),
    ("A", "Dylan", "就去湖边吧，听说那边新开了一家咖啡馆。"),
    ("B", "Vivian", "太好了，我一直想去试试他们家的拿铁。"),
    ("A", "Dylan", "那就这么说定了，下午两点在湖边见。"),
    ("B", "Vivian", "好的，不见不散！记得带上你那本没看完的书。"),
]

# 场景2 aaaabbbb：A先说4句，B再说4句（取前4轮）
DIALOGUE_AAAA = [
    ("A", "Dylan", "今天天气真不错，我们一起去公园散步吧。"),
    ("A", "Dylan", "就去湖边吧，听说那边新开了一家咖啡馆。"),
    ("A", "Dylan", "那就这么说定了，下午两点在湖边见。"),
    ("A", "Dylan", "对了，你上次说的那本书我看完了，确实很精彩。"),
    ("B", "Vivian", "好啊，我正好也想出去走走。你想去哪里？"),
    ("B", "Vivian", "太好了，我一直想去试试他们家的拿铁。"),
    ("B", "Vivian", "好的，不见不散！记得带上你那本没看完的书。"),
    ("B", "Vivian", "我也刚看完，有些地方还想跟你讨论一下呢。"),
]


def submit_and_wait(text, speaker, instruct="愉快，轻松，语速中等"):
    r = client.submit(text=text, language="Chinese", speaker=speaker, instruct=instruct)
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


def basic_features(data, label):
    """提取基础特征：时长、RMS、过零率"""
    if len(data) < 44:
        return {"label": label, "error": "数据太短"}

    channels = struct.unpack('<H', data[22:24])[0]
    sample_rate = struct.unpack('<I', data[24:28])[0]
    bits = struct.unpack('<H', data[34:36])[0]

    # 找 data chunk
    pos = 12
    pcm = None
    while pos < len(data) - 8:
        cid = data[pos:pos + 4]
        sz = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        if cid == b'data':
            pcm = data[pos + 8:pos + 8 + sz]
            break
        pos += 8 + sz
        if sz % 2:
            pos += 1

    if pcm is None:
        return {"label": label, "error": "无PCM"}

    bps = bits // 8
    n = len(pcm) // bps
    samples = []
    if bits == 16:
        for i in range(n):
            samples.append(struct.unpack('<h', pcm[i*2:i*2+2])[0] / 32768.0)
    elif bits == 32:
        for i in range(n):
            samples.append(struct.unpack('<i', pcm[i*4:i*4+4])[0] / 2147483648.0)
    else:
        return {"label": label, "error": f"位深{bits}"}

    n = len(samples)
    dur = n / sample_rate
    rms = math.sqrt(sum(s*s for s in samples) / n)
    zcr = sum(1 for i in range(1, n) if (samples[i] >= 0) != (samples[i-1] >= 0)) / n

    # 基频估计（自相关）
    frame_len = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.01)
    f0s = []
    for start in range(0, n - frame_len, hop):
        frame = samples[start:start+frame_len]
        e = math.sqrt(sum(s*s for s in frame) / frame_len)
        if e < 0.01:
            continue
        min_lag = max(1, int(sample_rate / 500))
        max_lag = min(frame_len-1, int(sample_rate / 50))
        if max_lag <= min_lag:
            continue
        best_c, best_l = -1, 0
        for lag in range(min_lag, max_lag):
            c = sum(frame[i]*frame[i+lag] for i in range(frame_len-lag)) / (frame_len-lag)
            if c > best_c:
                best_c, best_l = c, lag
        if best_l > 0 and best_c > 0.2:
            f0s.append(sample_rate / best_l)
    f0_mean = sum(f0s)/len(f0s) if f0s else 0
    f0_std = math.sqrt(sum((f-f0_mean)**2 for f in f0s)/len(f0s)) if f0s else 0

    return {
        "label": label,
        "duration": round(dur, 2),
        "rms": round(rms, 5),
        "zcr": round(zcr, 5),
        "f0_mean": round(f0_mean, 1),
        "f0_std": round(f0_std, 1),
        "size": len(data),
        "md5": hashlib.md5(data).hexdigest(),
    }


def check_stability(features_list, role_name):
    """检查同一角色的音色稳定性"""
    sizes = [f["size"] for f in features_list]
    f0s = [f["f0_mean"] for f in features_list if f["f0_mean"] > 0]
    rmss = [f["rms"] for f in features_list]

    size_ratio = min(sizes) / max(sizes) if max(sizes) > 0 else 0
    size_ok = size_ratio > 0.7

    f0_cv = (sum((f-sum(f0s)/len(f0s))**2 for f in f0s)/len(f0s))**0.5 / (sum(f0s)/len(f0s)) if f0s and sum(f0s) > 0 else 0
    f0_ok = f0_cv < 0.2

    rms_cv = (sum((r-sum(rmss)/len(rmss))**2 for r in rmss)/len(rmss))**0.5 / (sum(rmss)/len(rmss)) if rmss and sum(rmss) > 0 else 0
    rms_ok = rms_cv < 0.3

    print(f"\n  [{role_name}] 稳定性:")
    print(f"    文件大小: {sizes} | 最小/最大={size_ratio:.1%} {'✅' if size_ok else '❌'}")
    print(f"    基频F0:   {[f'{f:.1f}' for f in f0s]} | CV={f0_cv:.1%} {'✅' if f0_ok else '❌'}")
    print(f"    RMS能量:  {[f'{r:.4f}' for r in rmss]} | CV={rms_cv:.1%} {'✅' if rms_ok else '❌'}")

    return size_ok and f0_ok and rms_ok


# ════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 7: 双角色音色稳定性测试")
print("=" * 60)

all_features = []

# ── 场景1: ababab ──────────────────────────────────────────
print("\n【场景1】交替对话 ababab")
print("-" * 40)

for i, (role, speaker, text) in enumerate(DIALOGUE):
    label = f"AB_{i+1}_{role}_{speaker}"
    print(f"  [{i+1}/6] {speaker}: {text[:20]}...", end=" ", flush=True)
    data, err = submit_and_wait(text, speaker)
    if err:
        print(f"❌ {err}")
        all_features.append({"label": label, "error": err})
        continue
    path = save_wav(data, f"{label}.wav")
    feats = basic_features(data, label)
    all_features.append(feats)
    print(f"✅ {feats['size']}B F0={feats['f0_mean']:.0f}Hz RMS={feats['rms']:.4f}")

# 分析场景1
dylan_ab = [f for f in all_features if "Dylan" in f.get("label","") and "error" not in f]
vivian_ab = [f for f in all_features if "Vivian" in f.get("label","") and "error" not in f]

dylan_ab_ok = check_stability(dylan_ab, "Dylan(ababab)")
vivian_ab_ok = check_stability(vivian_ab, "Vivian(ababab)")

# ── 场景2: aaaabbbb ────────────────────────────────────────
print("\n【场景2】连续对话 aaaabbbb")
print("-" * 40)

features_aaaa = []
features_bbbb = []

for i, (role, speaker, text) in enumerate(DIALOGUE_AAAA):
    group = "A" if role == "A" else "B"
    label = f"AABB_{i+1}_{group}_{speaker}"
    print(f"  [{i+1}/8] {speaker}: {text[:20]}...", end=" ", flush=True)
    data, err = submit_and_wait(text, speaker)
    if err:
        print(f"❌ {err}")
        continue
    path = save_wav(data, f"{label}.wav")
    feats = basic_features(data, label)
    if role == "A":
        features_aaaa.append(feats)
    else:
        features_bbbb.append(feats)
    print(f"✅ {feats['size']}B F0={feats['f0_mean']:.0f}Hz RMS={feats['rms']:.4f}")

dylan_aaaa_ok = check_stability(features_aaaa, "Dylan(aaaa)")
vivian_bbbb_ok = check_stability(features_bbbb, "Vivian(bbbb)")

# ════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 音色稳定性测试汇总")
print("=" * 60)
print(f"  Dylan  ababab:  {'✅ 稳定' if dylan_ab_ok else '❌ 波动'}")
print(f"  Vivian ababab:  {'✅ 稳定' if vivian_ab_ok else '❌ 波动'}")
print(f"  Dylan  aaaa:    {'✅ 稳定' if dylan_aaaa_ok else '❌ 波动'}")
print(f"  Vivian bbbb:    {'✅ 稳定' if vivian_bbbb_ok else '❌ 波动'}")

all_ok = dylan_ab_ok and vivian_ab_ok and dylan_aaaa_ok and vivian_bbbb_ok
if all_ok:
    print("\n  🎉 全部角色音色稳定！")
else:
    print("\n  ⚠️ 存在音色波动，需人工复核")

print(f"\n  音频输出: {TEST_OUT}")
print("=" * 60)
