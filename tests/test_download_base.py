"""
从 ModelScope 下载 Qwen3-TTS Base 模型并验证可用性。

用法：
    python test_download_base.py

下载完成后会打印模型路径，替换到 config.yaml 的 model.model_path 即可。
"""
import sys
import os

# ── 1. 下载模型 ──────────────────────────────────────────
print("=" * 60)
print("Step 1: 从 ModelScope 下载 Qwen3-TTS-12Hz-1.7B-Base")
print("=" * 60)

try:
    from modelscope import snapshot_download
    model_dir = snapshot_download(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        cache_dir=None,
    )
    print(f"✅ 模型下载完成: {model_dir}")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    sys.exit(1)

# ── 2. 检查模型文件完整性 ─────────────────────────────────
print()
print("=" * 60)
print("Step 2: 检查模型文件")
print("=" * 60)

required_files = [
    "config.json",
    "tokenizer_config.json",
]
weight_files = [
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
]

has_weights = False
for f in weight_files:
    path = os.path.join(model_dir, f)
    exists = os.path.exists(path)
    print(f"  {'✅' if exists else '❌'} {f}")
    if exists:
        has_weights = True

for f in required_files:
    path = os.path.join(model_dir, f)
    exists = os.path.exists(path)
    print(f"  {'✅' if exists else '❌'} {f}")

if not has_weights:
    print("\n❌ 缺少模型权重文件！尝试用 modelscope 的 download 命令手动下载：")
    print('   modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base')
    sys.exit(1)

# ── 3. 加载模型验证 ───────────────────────────────────────
print()
print("=" * 60)
print("Step 3: 加载模型验证")
print("=" * 60)

try:
    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"  加载模型: {model_dir}")
    print(f"  dtype: float32")
    print(f"  device: cpu")

    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cpu",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# ── 4. 测试 create_voice_clone_prompt ────────────────────
print()
print("=" * 60)
print("Step 4: 测试 create_voice_clone_prompt（需要参考音频）")
print("=" * 60)
print("  跳过（需要手动提供参考音频）")
print()
print("=" * 60)
print("全部检查通过！")
print("=" * 60)
print()
print("将以下路径填入 config.yaml 的 model.model_path：")
print(f"  {model_dir}")
