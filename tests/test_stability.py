"""
Test 7: 双角色音色稳定性测试
==============================
同一段对白，Dylan 和 Vivian 两个角色：
  场景1 ababab：交替对话，验证交叉对话中音色是否稳定
  场景2 aaaabbbb：各自连续对话，验证连续对话中音色是否稳定

音色分析：纯 Python stdlib 实现 MFCC（不依赖 numpy/librosa）
  - 预加重 → 分帧 → 加窗 → DFT → 梅尔滤波器组 → 对数 → DCT → MFCC
  - 对比同一角色多次生成的 MFCC 余弦相似度
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

# ── 对白数据 ─────────────────────────────────────────────────
DIALOGUE = [
    ("A", "dylan", "今天天气真不错，我们一起去公园散步吧。"),
    ("B", "vivian", "好啊，我正好也想出去走走。你想去哪里？"),
    ("A", "dylan", "就去湖边吧，听说那边新开了一家咖啡馆。"),
    ("B", "vivian", "太好了，我一直想去试试他们家的拿铁。"),
    ("A", "dylan", "那就这么说定了，下午两点在湖边见。"),
    ("B", "vivian", "好的，不见不散！记得带上你那本没看完的书。"),
]

DIALOGUE_AAAA = [
    ("A", "dylan", "今天天气真不错，我们一起去公园散步吧。"),
    ("A", "dylan", "就去湖边吧，听说那边新开了一家咖啡馆。"),
    ("A", "dylan", "那就这么说定了，下午两点在湖边见。"),
    ("A", "dylan", "对了，你上次说的那本书我看完了，确实很精彩。"),
    ("B", "vivian", "好啊，我正好也想出去走走。你想去哪里？"),
    ("B", "vivian", "太好了，我一直想去试试他们家的拿铁。"),
    ("B", "vivian", "好的，不见不散！记得带上你那本没看完的书。"),
    ("B", "vivian", "我也刚看完，有些地方还想跟你讨论一下呢。"),
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


# ── WAV 解析 ─────────────────────────────────────────────────
def parse_wav(data):
    """解析 WAV，返回 (samples, sample_rate)"""
    if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        return None, 0
    channels = struct.unpack('<H', data[22:24])[0]
    sr = struct.unpack('<I', data[24:28])[0]
    bits = struct.unpack('<H', data[34:36])[0]
    pos = 12
    pcm = None
    while pos < len(data) - 8:
        cid = data[pos:pos+4]
        sz = struct.unpack('<I', data[pos+4:pos+8])[0]
        if cid == b'data':
            pcm = data[pos+8:pos+8+sz]
            break
        pos += 8 + sz
        if sz % 2: pos += 1
    if pcm is None:
        return None, 0
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
        return None, 0
    # 多声道取平均
    if channels > 1:
        merged = []
        for i in range(0, len(samples), channels):
            merged.append(sum(samples[i:i+channels]) / channels)
        samples = merged
    return samples, sr


# ── 纯 Python MFCC ───────────────────────────────────────────
def hz_to_mel(hz):
    return 2595 * math.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def pre_emphasis(samples, coeff=0.97):
    """预加重"""
    result = [samples[0]]
    for i in range(1, len(samples)):
        result.append(samples[i] - coeff * samples[i-1])
    return result

def framing(samples, sr, frame_size=0.025, frame_stride=0.01):
    """分帧"""
    frame_len = int(sr * frame_size)
    step = int(sr * frame_stride)
    frames = []
    for start in range(0, len(samples) - frame_len, step):
        frames.append(samples[start:start+frame_len])
    return frames

def window(frame):
    """汉明窗"""
    n = len(frame)
    return [frame[i] * (0.54 - 0.46 * math.cos(2 * math.pi * i / (n-1))) for i in range(n)]

def dft(frame):
    """离散傅里叶变换（返回幅度谱），只取前半部分"""
    n = len(frame)
    half = n // 2 + 1
    mag = []
    for k in range(half):
        real = sum(frame[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
        imag = sum(frame[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
        mag.append(math.sqrt(real*real + imag*imag))
    return mag

def mel_filterbank(num_filters, fft_size, sr):
    """梅尔滤波器组"""
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    # 在梅尔尺度上均匀分布的点
    points = [mel_to_hz(low_mel + i * (high_mel - low_mel) / (num_filters + 1))
              for i in range(num_filters + 2)]
    # 转为 FFT bin
    bins = [int((fft_size + 1) * p / sr) for p in points]
    filters = []
    for i in range(num_filters):
        f = [0.0] * (fft_size // 2 + 1)
        for j in range(bins[i], bins[i+1]):
            if bins[i+1] != bins[i]:
                f[j] = (j - bins[i]) / (bins[i+1] - bins[i])
        for j in range(bins[i+1], bins[i+2]):
            if bins[i+2] != bins[i+1]:
                f[j] = (bins[i+2] - j) / (bins[i+2] - bins[i+1])
        filters.append(f)
    return filters

def apply_filterbank(mag_spec, filters):
    """应用滤波器组"""
    energies = []
    for f in filters:
        e = sum(mag_spec[j] * f[j] for j in range(min(len(mag_spec), len(f))))
        energies.append(max(e, 1e-10))  # 防止 log(0)
    return energies

def dct(energies, num_ceps=13):
    """离散余弦变换，提取前 num_ceps 个系数"""
    n = len(energies)
    coeffs = []
    for k in range(num_ceps):
        c = sum(energies[i] * math.cos(math.pi * k * (2*i + 1) / (2 * n)) for i in range(n))
        coeffs.append(c)
    return coeffs

def compute_mfcc(samples, sr, num_filters=26, num_ceps=13):
    """计算 MFCC，返回 13 维系数向量"""
    # 预加重
    emphasized = pre_emphasis(samples)
    # 分帧
    frames = framing(emphasized, sr)
    if not frames:
        return [0.0] * num_ceps
    # 加窗 + DFT + 滤波器组 + 对数 + DCT
    fb = mel_filterbank(num_filters, len(frames[0]), sr)
    all_ceps = []
    for frame in frames:
        w = window(frame)
        mag = dft(w)
        energies = apply_filterbank(mag, fb)
        log_e = [math.log(e) for e in energies]
        ceps = dct(log_e, num_ceps)
        all_ceps.append(ceps)
    # 取所有帧的均值作为整段音频的 MFCC 特征
    n_frames = len(all_ceps)
    if n_frames == 0:
        return [0.0] * num_ceps
    mean_ceps = []
    for k in range(num_ceps):
        mean_ceps.append(sum(all_ceps[i][k] for i in range(n_frames)) / n_frames)
    return mean_ceps


# ── 余弦相似度 ──────────────────────────────────────────────
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


# ── 特征提取（封装）────────────────────────────────────────────
def extract_features(data, label):
    """提取 MFCC + 基础特征"""
    samples, sr = parse_wav(data)
    if samples is None:
        return {"label": label, "error": "WAV解析失败"}

    n = len(samples)
    dur = n / sr
    rms = math.sqrt(sum(s*s for s in samples) / n)

    # MFCC（核心音色特征）
    mfcc = compute_mfcc(samples, sr)

    return {
        "label": label,
        "duration": round(dur, 2),
        "rms": round(rms, 5),
        "mfcc": mfcc,
        "size": len(data),
        "md5": hashlib.md5(data).hexdigest(),
    }


def check_stability(features_list, role_name):
    """音色稳定性分析"""
    n = len(features_list)
    if n < 2:
        print(f"\n  [{role_name}] 结果不足({n}个)，跳过")
        return False

    print(f"\n  [{role_name}] 音色稳定性 ({n}段):")

    # MFCC 两两余弦相似度
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = cosine_similarity(features_list[i]["mfcc"], features_list[j]["mfcc"])
            sims.append(sim)
            print(f"    {features_list[i]['label']} ↔ {features_list[j]['label']}: {sim:.3f}")

    avg_sim = sum(sims) / len(sims) if sims else 0
    min_sim = min(sims) if sims else 0
    mfcc_ok = avg_sim > 0.85

    print(f"    平均相似度: {avg_sim:.3f} | 最低: {min_sim:.3f} | {'✅' if mfcc_ok else '❌'}")

    # RMS 稳定性
    rmss = [f["rms"] for f in features_list]
    rms_avg = sum(rmss) / n
    rms_cv = math.sqrt(sum((r-rms_avg)**2 for r in rmss)/n) / rms_avg if rms_avg > 0 else 0
    rms_ok = rms_cv < 0.3
    print(f"    RMS: {[f'{r:.4f}' for r in rmss]} | CV={rms_cv:.1%} | {'✅' if rms_ok else '❌'}")

    return mfcc_ok and rms_ok


# ════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 7: 双角色音色稳定性测试 (纯Python MFCC)")
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
    feats = extract_features(data, label)
    all_features.append(feats)
    if "error" not in feats:
        print(f"✅ {feats['size']}B rms={feats['rms']:.4f} mfcc[0]={feats['mfcc'][0]:.2f}")
    else:
        print(f"❌ {feats['error']}")

dylan_ab = [f for f in all_features if "dylan" in f.get("label","") and "error" not in f]
vivian_ab = [f for f in all_features if "vivian" in f.get("label","") and "error" not in f]

dylan_ab_ok = check_stability(dylan_ab, "dylan(ababab)")
vivian_ab_ok = check_stability(vivian_ab, "vivian(ababab)")

# ── 场景2: aaaabbbb ────────────────────────────────────────
print("\n【场景2】连续对话 aaaabbbb")
print("-" * 40)

features_aaaa = []
features_bbbb = []

for i, (role, speaker, text) in enumerate(DIALOGUE_AAAA):
    label = f"AABB_{i+1}_{role}_{speaker}"
    print(f"  [{i+1}/8] {speaker}: {text[:20]}...", end=" ", flush=True)
    data, err = submit_and_wait(text, speaker)
    if err:
        print(f"❌ {err}")
        continue
    path = save_wav(data, f"{label}.wav")
    feats = extract_features(data, label)
    if "error" not in feats:
        if role == "A":
            features_aaaa.append(feats)
        else:
            features_bbbb.append(feats)
        print(f"✅ {feats['size']}B rms={feats['rms']:.4f} mfcc[0]={feats['mfcc'][0]:.2f}")
    else:
        print(f"❌ {feats['error']}")

dylan_aaaa_ok = check_stability(features_aaaa, "dylan(aaaa)")
vivian_bbbb_ok = check_stability(features_bbbb, "vivian(bbbb)")

# ════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 音色稳定性测试汇总")
print("=" * 60)
print(f"  dylan  ababab:  {'✅ 稳定' if dylan_ab_ok else '❌ 波动'}")
print(f"  vivian ababab:  {'✅ 稳定' if vivian_ab_ok else '❌ 波动'}")
print(f"  dylan  aaaa:    {'✅ 稳定' if dylan_aaaa_ok else '❌ 波动'}")
print(f"  vivian bbbb:    {'✅ 稳定' if vivian_bbbb_ok else '❌ 波动'}")

all_ok = dylan_ab_ok and vivian_ab_ok and dylan_aaaa_ok and vivian_bbbb_ok
if all_ok:
    print("\n  🎉 全部角色音色稳定！")
else:
    print("\n  ⚠️ 存在音色波动，需人工复核")

print(f"\n  音频输出: {TEST_OUT}")
print("=" * 60)
