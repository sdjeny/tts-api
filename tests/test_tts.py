"""
三角色广播剧生成脚本
角色：旁白、主角（林轩）、配角（苏婉）
情感变化：喜、怒、哀、乐，但角色声音识别不丢失

用法：先启动 server.py，再运行此脚本
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_client import TtsClient

# ── 客户端 ─────────────────────────────────────────────────
client = TtsClient.from_config(
    config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
)

# ── 角色配置 ───────────────────────────────────────────────
role_to_speaker = {
    "旁白": "Uncle_Fu",
    "主角_林轩": "Dylan",
    "配角_苏婉": "Vivian",
}

CHARACTER_VOICES = {
    "旁白": {
        "voice_description": (
            "一位四十岁左右的男性专业旁白，声音沉稳、清晰、富有磁性。"
            "语调客观，不带个人情绪，像纪录片解说那样娓娓道来。"
            "语速中等偏慢，每个字都咬得很清楚。"
        )
    },
    "主角_林轩": {
        "voice_description": (
            "一位二十五岁的年轻男性，声音温暖有朝气，略带一点少年感。"
            "音色明亮，底气充足，说话时带着年轻人特有的活力。"
            "即使情绪变化，音色本身的质感保持不变。"
        )
    },
    "配角_苏婉": {
        "voice_description": (
            "一位二十三岁的年轻女性，声音清甜柔和，听感舒适。"
            "音色像春风一样轻软，但不失清晰度。"
            "即使愤怒或悲伤，声线的甜美本质也不会改变。"
        )
    },
}

# ── 剧本 ───────────────────────────────────────────────────
script = [
    ("旁白", "夜幕降临，城市的灯火渐渐亮起。林轩独自站在天台上，望着远处出神。", "", "Chinese"),
    ("旁白", "他刚刚经历了一场激烈的争吵，心里五味杂陈。", "", "Chinese"),
    ("主角_林轩", "他们凭什么这样对我？我辛辛苦苦这么久，一句话就把我打发了？", "愤怒，语气强烈，语速稍快", "Chinese"),
    ("配角_苏婉", "林轩，你先冷静下来。我知道你很难过，但生气解决不了问题。", "温柔，关切，语速平缓", "Chinese"),
    ("主角_林轩", "不只是生气……我更多的是失望。我一直相信他们，没想到结果会是这样。", "悲伤，低落，语气沉重", "Chinese"),
    ("配角_苏婉", "失望是因为你在乎，这没有错。但你不能让这一次失败否定掉你的全部。", "温暖，鼓励，略带坚定", "Chinese"),
    ("旁白", "苏婉的话像一股暖流，缓缓流入林轩的心里。他沉默了片刻。", "", "Chinese"),
    ("主角_林轩", "你说的对。或许……是我太钻牛角尖了。只是，我真的不知道接下来该怎么走。", "语气缓和，带着一丝迷茫", "Chinese"),
    ("配角_苏婉", "还记得我们大学时一起做的那个项目吗？当时所有人都觉得我们不行，可最后呢？", "语气轻快，带着笑意，鼓励", "Chinese"),
    ("主角_林轩", "哈哈，那次啊……我们熬了整整三个通宵，你还差点把咖啡泼在电源上。", "开心，笑，放松", "Chinese"),
    ("配角_苏婉", "那就是青春啊！所以你看，摔倒了再爬起来，你从来都不是一个人。", "调皮，活泼，语速稍快", "Chinese"),
    ("旁白", "天台上，两个人的笑声冲淡了夜晚的寒意。有时候，一句简单的理解，就足以让阴霾散去。", "", "Chinese"),
    ("主角_林轩", "谢谢你，苏婉。我想我找到重新开始的方向了。明天，我会让他们看到不一样的我。", "坚定，充满希望，语速中等", "Chinese"),
    ("配角_苏婉", "这才是我认识的林轩。走吧，我请你喝杯热奶茶，庆祝你的新生。", "欣慰，温暖，带着小小的骄傲", "Chinese"),
    ("旁白", "灯光下，两个年轻人并肩走下天台，融进了城市的万家灯火中。", "", "Chinese"),
]

# ── 输出目录 ───────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "radio_drama")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 提交所有任务 ───────────────────────────────────────────
print("=" * 60)
print("提交广播剧任务")
print("=" * 60)

tasks = []  # (idx, role, task_id)
for idx, (role, text, instruct, lang) in enumerate(script):
    speaker = role_to_speaker[role]
    r = client.submit(text=text, language=lang, speaker=speaker, instruct=instruct)
    if r.error:
        print(f"  ❌ [{idx+1}/{len(script)}] {role} 提交失败: {r.error}")
        tasks.append((idx, role, None))
    else:
        print(f"  [{idx+1}/{len(script)}] {role} → {r.task_id}")
        tasks.append((idx, role, r.task_id))

# ── 等待并下载 ─────────────────────────────────────────────
print()
print("=" * 60)
print("等待生成并下载")
print("=" * 60)

for idx, role, task_id in tasks:
    if task_id is None:
        print(f"  ⚠️  跳过 {role}（提交失败）")
        continue

    sr = client.wait(task_id)
    if not sr.ok:
        print(f"  ❌ {role} 任务失败: {sr.status} {sr.error}")
        continue

    dl = client.download(task_id)
    if not dl.ok:
        print(f"  ❌ {role} 下载失败: {dl.error}")
        continue

    emotion_tag = script[idx][2] if script[idx][2] else "中性"
    safe_emotion = "".join(c for c in emotion_tag if c.isalnum() or c in (' ', '_', '-', '，'))
    filename = f"{idx+1:02d}_{role}_{safe_emotion}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(dl.data)
    print(f"  ✅ {role} → {filename} ({len(dl.data)} 字节)")

print()
print(f"✅ 全部完成！共 {len(script)} 句，保存在 '{OUTPUT_DIR}/'")
