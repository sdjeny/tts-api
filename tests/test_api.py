"""
TTS API 接口测试
用法：先启动 server.py，再运行此脚本

测试分层：
  - TestTTSAPILegacy   基础接口回归测试
  - TestTTSAPISampling  采样参数验证
"""
import os
import unittest
import sys

# 确保能 import tts_client（从项目根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_client import TtsClient

client = TtsClient.from_config(config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
BASE_URL = client.base_url

# 测试输出目录
TEST_OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")


# ============================================================
# 基础接口回归测试
# ============================================================

class TestTTSAPILegacy(unittest.TestCase):
    """基础 API 兼容性测试"""

    def test_01_health(self):
        """健康检查"""
        self.assertTrue(client.health())
        print("[OK] health ok")

    def test_02_submit(self):
        """提交任务（带 speaker 和 instruct），返回 task_id"""
        result = client.submit(
            text="你好，欢迎使用语音合成服务。今天天气真不错，我们一起去公园散步吧。",
            language="Chinese",
            speaker="Dylan",
            instruct="愉快，轻松，语速中等",
        )
        self.assertFalse(result.error, f"提交失败: {result.error}")
        self.assertRegex(result.task_id, r"^\d{8}_[0-9a-f]+$")
        self.__class__.legacy_task_id = result.task_id
        print(f"[OK] 提交成功，task_id={result.task_id}，排队位置={result.position}")

    def test_03_submit_no_text(self):
        """空文本应返回错误"""
        result = client.submit(text="")
        self.assertTrue(result.error)
        print("[OK] 空文本校验通过")

    def test_04_status_pending_or_processing(self):
        """查询刚提交的任务状态"""
        sr = client.status(self.legacy_task_id)
        self.assertIn(sr.status, ("pending", "processing", "success"))
        print(f"[OK] 状态查询: {sr.status}")

    def test_05_status_not_found(self):
        """不存在的 task_id 返回错误"""
        sr = client.status("no_such_id")
        self.assertTrue(sr.error)
        print("[OK] 不存在任务校验通过")

    def test_06_queue_info(self):
        """查看队列概况"""
        info = client.queue_info()
        self.assertIn("queue_size", info)
        self.assertIn("tasks_tracked", info)
        print(f"[OK] 队列概况: {info}")

    def test_07_wait_for_success(self):
        """等待任务完成，验证最终状态为 success"""
        sr = client.wait(self.legacy_task_id)
        self.assertTrue(sr.ok, f"任务未完成: {sr.status} {sr.error}")
        print(f"[OK] 任务最终状态: {sr.status}")

    def test_08_download(self):
        """下载生成的音频文件"""
        dl = client.download(self.legacy_task_id)
        self.assertTrue(dl.ok, f"下载失败: {dl.error}")
        self.assertGreater(len(dl.data), 1000)

        os.makedirs(TEST_OUT_DIR, exist_ok=True)
        out_path = os.path.join(TEST_OUT_DIR, "legacy_download.wav")
        with open(out_path, "wb") as f:
            f.write(dl.data)
        print(f"[OK] 下载成功，{len(dl.data)} 字节，已保存到 {out_path}")

    def test_09_download_404(self):
        """下载不存在的任务应返回错误"""
        dl = client.download("no_such_id")
        self.assertFalse(dl.ok)
        print("[OK] 下载不存在任务校验通过")

    def test_10_multi_role_emotion(self):
        """多角色情感测试：旁白 + 愤怒 + 温柔"""
        cases = [
            ("夜幕降临，城市的灯火渐渐亮起。林轩独自站在天台上，望着远处出神。",
             "Uncle_Fu", "沉稳，客观，语速中等偏慢"),
            ("他们凭什么这样对我？我辛辛苦苦这么久，一句话就把我打发了？",
             "Dylan", "愤怒，语气强烈，语速稍快"),
            ("林轩，你先冷静下来。我知道你很难过，但生气解决不了问题。",
             "Vivian", "温柔，关切，语速平缓"),
        ]
        task_ids = []
        for text, speaker, instruct in cases:
            r = client.submit(text=text, speaker=speaker, instruct=instruct)
            self.assertFalse(r.error)
            task_ids.append(r.task_id)
            print(f"  提交 {speaker}({instruct}) → {r.task_id}")

        for i, tid in enumerate(task_ids):
            sr = client.wait(tid)
            self.assertTrue(sr.ok, f"任务 {tid} 未完成: {sr.status}")
            dl = client.download(tid)
            self.assertTrue(dl.ok, f"下载 {tid} 失败: {dl.error}")
            self.assertGreater(len(dl.data), 1000)
            os.makedirs(TEST_OUT_DIR, exist_ok=True)
            out_path = os.path.join(TEST_OUT_DIR, f"legacy_role_{i}.wav")
            with open(out_path, "wb") as f:
                f.write(dl.data)
            print(f"  [OK] {cases[i][1]} 下载成功，{len(dl.data)} 字节")

        print("[OK] 多角色情感测试全部通过")


# ============================================================
# 采样参数验证
# ============================================================

class TestTTSAPISampling(unittest.TestCase):
    """验证服务端采样参数透传"""

    SAMPLE_TEXT = "Today is a nice day. Let's go for a walk in the park. What do you think?"

    def _submit_and_wait(self, **kwargs):
        """提交任务 + 等待完成 + 下载，返回 (task_id, audio_bytes)"""
        r = client.submit(text=self.SAMPLE_TEXT, **kwargs)
        self.assertFalse(r.error, f"提交失败: {r.error}")
        task_id = r.task_id
        print(f"  提交 → {task_id}")

        sr = client.wait(task_id)
        self.assertTrue(sr.ok, f"任务未完成: {sr.status} {sr.error}")

        dl = client.download(task_id)
        self.assertTrue(dl.ok, f"下载失败: {dl.error}")
        self.assertGreater(len(dl.data), 1000)

        return task_id, dl.data

    def _save_wav(self, data: bytes, filename: str):
        os.makedirs(TEST_OUT_DIR, exist_ok=True)
        out_path = os.path.join(TEST_OUT_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path

    def test_01_default_params(self):
        """不传采样参数 → 服务端使用保守默认值"""
        tid, data = self._submit_and_wait(speaker="Dylan", instruct="愉快，轻松")
        path = self._save_wav(data, "sampling_default.wav")
        print(f"[OK] 默认参数音频: {path} ({len(data)} 字节)")

    def test_02_conservative_params(self):
        """显式保守参数"""
        tid, data = self._submit_and_wait(
            speaker="Dylan", instruct="愉快，轻松",
            temperature=0.3, top_k=20, top_p=0.85, repetition_penalty=1.1,
        )
        path = self._save_wav(data, "sampling_conservative.wav")
        print(f"[OK] 保守参数音频: {path} ({len(data)} 字节)")

    def test_03_official_defaults(self):
        """官方默认参数"""
        tid, data = self._submit_and_wait(
            speaker="Dylan", instruct="愉快，轻松",
            temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05,
        )
        path = self._save_wav(data, "sampling_official.wav")
        print(f"[OK] 官方默认参数音频: {path} ({len(data)} 字节)")

    def test_04_ultra_stable(self):
        """极限稳定参数（贪心解码）"""
        tid, data = self._submit_and_wait(
            speaker="Dylan", instruct="愉快，轻松",
            temperature=0.1, do_sample=False, top_k=10, top_p=0.5, repetition_penalty=1.3,
        )
        path = self._save_wav(data, "sampling_ultra_stable.wav")
        print(f"[OK] 极限稳定参数音频: {path} ({len(data)} 字节)")

    def test_05_partial_params(self):
        """仅传部分参数，其余 None 降级"""
        tid, data = self._submit_and_wait(
            speaker="Dylan", instruct="愉快，轻松",
            temperature=0.5,
        )
        path = self._save_wav(data, "sampling_partial.wav")
        print(f"[OK] 部分参数音频: {path} ({len(data)} 字节)")

    def test_06_consistency_same_text(self):
        """同一文本 + 保守参数，连续生成 3 次，大小差异 <30%"""
        sizes = []
        for i in range(3):
            _, data = self._submit_and_wait(
                speaker="Dylan", instruct="愉快，轻松",
                temperature=0.3, top_k=20, top_p=0.85, repetition_penalty=1.1,
            )
            self._save_wav(data, f"consistency_{i}.wav")
            sizes.append(len(data))
            print(f"  第 {i+1} 次: {len(data)} 字节")

        ratio = min(sizes) / max(sizes) if max(sizes) > 0 else 0
        self.assertGreater(ratio, 0.7, f"三次生成大小差异过大: {sizes}")
        print(f"[OK] 一致性测试通过，大小范围 {min(sizes)}~{max(sizes)} 字节，最小/最大={ratio:.2%}")

    def test_07_cross_sentence_stability(self):
        """不同文本 + 保守参数，跨句稳定性"""
        texts = [
            "今天天气真不错，我们一起去公园散步吧。",
            "你知道吗？我昨天做了一个特别奇怪的梦。",
            "这本书我已经读了三遍了，每次都有新的收获。",
        ]
        sizes = []
        for i, text in enumerate(texts):
            r = client.submit(
                text=text, speaker="Dylan", instruct="愉快，轻松",
                temperature=0.3, top_k=20, top_p=0.85, repetition_penalty=1.1,
            )
            self.assertFalse(r.error)
            sr = client.wait(r.task_id)
            self.assertTrue(sr.ok)
            dl = client.download(r.task_id)
            self.assertTrue(dl.ok)
            self._save_wav(dl.data, f"cross_sentence_{i}.wav")
            sizes.append(len(dl.data))
            print(f"  句子 {i+1}: {len(dl.data)} 字节")

        print(f"[OK] 跨句稳定性测试完成，大小范围 {min(sizes)}~{max(sizes)} 字节")


if __name__ == "__main__":
    print("=" * 60)
    print("TTS API Auto Test")
    print(f"Server: {BASE_URL}")
    print("=" * 60)
    print()
    unittest.main(verbosity=2)
