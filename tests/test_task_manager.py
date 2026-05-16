"""
Tests for task_manager.py — SQLite 持久化任务管理器
====================================================
Run with: python -m pytest tests/test_task_manager.py -v
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from task_manager import TaskManager, STATUS_PENDING, STATUS_PROCESSING, \
    STATUS_SUCCESS, STATUS_FAILED, STATUS_CANCELLED


@pytest.fixture
def tm():
    """Create a TaskManager with a temp database for each test."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_tasks.db")
    manager = TaskManager(db_path)
    yield manager
    manager.close()
    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestTaskManager:
    """Test suite for TaskManager."""

    def test_submit_and_get(self, tm):
        """提交任务后能正确获取"""
        task_id = tm.submit("你好世界", extra={"language": "Chinese", "speaker": "af_heart"})
        assert task_id is not None
        assert task_id.startswith("202")  # starts with date prefix

        task = tm.get(task_id)
        assert task is not None
        assert task["text"] == "你好世界"
        assert task["status"] == STATUS_PENDING
        assert task["extra"]["language"] == "Chinese"
        assert task["extra"]["speaker"] == "af_heart"

    def test_submit_default_extra(self, tm):
        """不传 extra 时默认空 dict"""
        task_id = tm.submit("测试文本")
        task = tm.get(task_id)
        assert task["extra"] == {}

    def test_get_nonexistent(self, tm):
        """查询不存在的任务返回 None"""
        assert tm.get("nonexistent_id") is None

    def test_list_all(self, tm):
        """列表返回所有任务"""
        tm.submit("任务1")
        tm.submit("任务2")
        result = tm.list()
        assert result["total"] == 2
        assert len(result["tasks"]) == 2

    def test_list_with_status_filter(self, tm):
        """按状态筛选"""
        id1 = tm.submit("任务1")
        tm.update_status(id1, STATUS_SUCCESS)
        id2 = tm.submit("任务2")

        result = tm.list(status=STATUS_PENDING)
        assert result["total"] == 1
        assert result["tasks"][0]["task_id"] == id2

        result = tm.list(status=STATUS_SUCCESS)
        assert result["total"] == 1
        assert result["tasks"][0]["task_id"] == id1

    def test_list_with_search(self, tm):
        """按文本搜索"""
        tm.submit("hello world")
        tm.submit("你好世界")
        tm.submit("hello你好")

        result = tm.list(search="hello")
        assert result["total"] == 2

        result = tm.list(search="世界")
        assert result["total"] == 1

    def test_list_pagination(self, tm):
        """分页测试"""
        for i in range(25):
            tm.submit(f"任务{i}")
        result = tm.list(limit=10, offset=0)
        assert len(result["tasks"]) == 10
        assert result["total"] == 25
        assert result["offset"] == 0
        assert result["limit"] == 10

        result2 = tm.list(limit=10, offset=10)
        assert len(result2["tasks"]) == 10
        assert result2["offset"] == 10

    def test_update_status(self, tm):
        """更新任务状态"""
        task_id = tm.submit("测试")
        tm.update_status(task_id, STATUS_PROCESSING, started_at="2026-01-01T00:00:00")
        task = tm.get(task_id)
        assert task["status"] == STATUS_PROCESSING
        assert task["started_at"] == "2026-01-01T00:00:00"

        tm.update_status(task_id, STATUS_SUCCESS,
                         file_path="/tmp/test.wav",
                         finished_at="2026-01-01T00:01:00")
        task = tm.get(task_id)
        assert task["status"] == STATUS_SUCCESS
        assert task["file_path"] == "/tmp/test.wav"

    def test_cancel_pending(self, tm):
        """取消 pending 任务"""
        task_id = tm.submit("测试")
        ok = tm.cancel(task_id)
        assert ok is True
        task = tm.get(task_id)
        assert task["status"] == STATUS_CANCELLED

    def test_cancel_success_fails(self, tm):
        """已完成任务不可取消"""
        task_id = tm.submit("测试")
        tm.update_status(task_id, STATUS_SUCCESS)
        ok = tm.cancel(task_id)
        assert ok is False

    def test_delete(self, tm):
        """删除任务"""
        task_id = tm.submit("测试")
        ok = tm.delete(task_id, delete_file=False)
        assert ok is True
        assert tm.get(task_id) is None

    def test_delete_nonexistent(self, tm):
        """删除不存在的任务返回 False"""
        ok = tm.delete("nonexistent")
        assert ok is False

    def test_set_priority(self, tm):
        """设置优先级"""
        task_id = tm.submit("测试")
        ok = tm.set_priority(task_id, 5)
        assert ok is True
        task = tm.get(task_id)
        assert task["priority"] == 5

    def test_priority_ordering(self, tm):
        """高优先级任务排在前面"""
        id1 = tm.submit("低优先级", priority=0)
        id2 = tm.submit("高优先级", priority=10)
        id3 = tm.submit("中优先级", priority=5)

        result = tm.list()
        # Should be ordered by priority DESC
        assert result["tasks"][0]["task_id"] == id2  # priority 10
        assert result["tasks"][1]["task_id"] == id3  # priority 5
        assert result["tasks"][2]["task_id"] == id1  # priority 0

    def test_retry(self, tm):
        """重试失败任务"""
        task_id = tm.submit("测试")
        tm.update_status(task_id, STATUS_FAILED, error="测试错误")
        ok = tm.retry(task_id)
        assert ok is True
        task = tm.get(task_id)
        assert task["status"] == STATUS_PENDING
        assert task["error"] is None
        assert task["started_at"] is None
        assert task["finished_at"] is None

    def test_retry_non_failed(self, tm):
        """非失败任务不可重试"""
        task_id = tm.submit("测试")
        ok = tm.retry(task_id)
        assert ok is False

    def test_stats(self, tm):
        """统计信息"""
        id1 = tm.submit("任务1")
        id2 = tm.submit("任务2")
        tm.update_status(id1, STATUS_SUCCESS, finished_at="2026-01-01T00:01:00",
                         started_at="2026-01-01T00:00:00")

        stats = tm.stats()
        assert stats["total"] == 2
        assert stats["by_status"][STATUS_PENDING] == 1
        assert stats["by_status"][STATUS_SUCCESS] == 1
        assert stats["queue_depth"] == 1

    def test_cleanup(self, tm):
        """清理旧任务"""
        # Create old tasks (we can't easily set submitted_at, so cleanup with days=0)
        id1 = tm.submit("旧任务1")
        id2 = tm.submit("旧任务2")
        tm.update_status(id1, STATUS_SUCCESS, finished_at="2020-01-01T00:00:00",
                         started_at="2020-01-01T00:00:00",
                         file_path="/tmp/nonexistent.wav")

        # Manually set old submitted_at via raw SQL
        conn = tm._get_conn()
        conn.execute("UPDATE tasks SET submitted_at='2020-01-01T00:00:00' WHERE task_id=?", (id1,))
        conn.commit()

        deleted = tm.cleanup(days=0, delete_file=False)
        assert deleted >= 1  # id1 should be cleaned up
        assert tm.get(id1) is None
        assert tm.get(id2) is not None  # new task should remain

    def test_concurrent_submit(self, tm):
        """并发提交不会 crash"""
        import threading
        results = []
        def submit_worker(n):
            for i in range(5):
                tid = tm.submit(f"并发任务{n}-{i}")
                results.append(tid)

        threads = [threading.Thread(target=submit_worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 15
        assert tm.list()["total"] == 15

    def test_extra_serialization(self, tm):
        """extra 字段 JSON 序列化/反序列化正确"""
        extra = {
            "language": "Chinese",
            "speaker": "dylan",
            "instruct": "愉快",
            "temperature": 0.5,
            "do_sample": True,
            "top_k": 20,
            "top_p": 0.85,
            "speed": 1.0,
        }
        task_id = tm.submit("测试文本", extra=extra)
        task = tm.get(task_id)
        assert task["extra"] == extra
        # Verify JSON stored in DB
        conn = tm._get_conn()
        row = conn.execute("SELECT extra FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        raw = json.loads(row["extra"])
        assert raw == extra

    def test_qsize(self, tm):
        """队列大小正确"""
        assert tm.qsize() == 0
        tm.submit("任务1")
        assert tm.qsize() == 1
        tm.submit("任务2")
        assert tm.qsize() == 2

    def test_get_pending_task(self, tm):
        """从 pending 队列取任务"""
        id1 = tm.submit("任务1")
        id2 = tm.submit("任务2")

        task = tm.get_pending_task(block=False)
        assert task is not None
        assert task["task_id"] == id1  # FIFO order

        task = tm.get_pending_task(block=False)
        assert task["task_id"] == id2

        # Queue should be empty now
        task = tm.get_pending_task(block=False, timeout=0.1)
        assert task is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])