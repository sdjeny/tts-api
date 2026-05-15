"""
task_manager.py — SQLite 持久化任务管理器
============================================
替换 server.py 中原有的内存 task_queue + task_store + store_lock。

设计要点：
- WAL 模式 + check_same_thread=False 支持多线程并发读写
- extra 字段存 JSON（language, speaker, instruct, temperature 等 handler 特有参数）
- 优先级队列：按 priority DESC, submitted_at ASC 排序
- 所有方法线程安全（SQLite 自身 + Python Lock 双重保护）
"""
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Optional

log = logging.getLogger("tts-api.task-manager")

# ── 默认配置 ──────────────────────────────────────────────
_DEFAULT_DB_PATH = "data/tasks.db"

# ── SQL 建表 ─────────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    status          TEXT DEFAULT 'pending',  -- pending|processing|success|failed|cancelled
    priority        INTEGER DEFAULT 0,
    submitted_at    TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    file_path       TEXT,
    error           TEXT,
    extra           TEXT DEFAULT '{}',
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority DESC, submitted_at ASC);
"""

# ── 任务状态常量 ──────────────────────────────────────────
STATUS_PENDING    = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS    = "success"
STATUS_FAILED     = "failed"
STATUS_CANCELLED  = "cancelled"

_TERMINAL_STATUSES = {STATUS_SUCCESS, STATUS_FAILED, STATUS_CANCELLED}


class TaskManager:
    """SQLite 持久化任务管理器，线程安全。"""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self._db_path = str(Path(db_path).resolve())
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

        # 内存队列：只跟踪 pending 任务，worker 从这里取
        self._pending_queue: Queue = Queue()
        self._pending_set: set = set()  # 跟踪哪些 task_id 已在队列中

        # 启动时从 SQLite 恢复未完成任务
        self._restore_pending()

    # ── 数据库连接管理 ────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接（WAL 模式）。"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_db(self):
        """初始化数据库和表。"""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        # execscript 每条语句分别执行，建表和索引
        for statement in _CREATE_TABLE_SQL.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        conn.commit()
        log.info("TaskManager DB initialized: %s", self._db_path)

    def close(self):
        """关闭当前线程的数据库连接。"""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # ── 恢复未完成任务 ────────────────────────────────────

    def _restore_pending(self):
        """启动时从 SQLite 恢复 pending 任务到内存队列。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM tasks WHERE status=? ORDER BY priority DESC, submitted_at ASC",
            (STATUS_PENDING,)
        ).fetchall()
        for row in rows:
            task = self._row_to_dict(row)
            self._pending_queue.put(task)
            self._pending_set.add(task["task_id"])
        if rows:
            log.info("Restored %d pending tasks from SQLite", len(rows))

    # ── 内部工具 ──────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """将 sqlite3.Row 转为 dict，并解析 extra 字段。"""
        d = dict(row)
        try:
            d["extra"] = json.loads(d.get("extra", "{}"))
        except (json.JSONDecodeError, TypeError):
            d["extra"] = {}
        return d

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _serialize_extra(extra: Optional[dict]) -> str:
        if extra is None:
            return "{}"
        return json.dumps(extra, ensure_ascii=False)

    # ── 核心 API ──────────────────────────────────────────

    def submit(self, text: str, extra: Optional[dict] = None,
               priority: int = 0) -> str:
        """
        提交新任务。
        - text: 必填，要合成的文本
        - extra: handler 特有参数 dict（language, speaker, instruct, temperature 等）
        - priority: 优先级（越大越优先，默认 0）
        返回 task_id。
        """
        import uuid
        date_prefix = datetime.now().strftime("%Y%m%d")
        task_id = f"{date_prefix}_{uuid.uuid4().hex}"

        task = {
            "task_id": task_id,
            "text": text,
            "status": STATUS_PENDING,
            "priority": priority,
            "submitted_at": self._now(),
            "started_at": None,
            "finished_at": None,
            "file_path": None,
            "error": None,
            "extra": self._serialize_extra(extra),
        }

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tasks
               (task_id, text, status, priority, submitted_at, extra)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (task["task_id"], task["text"], task["status"],
             task["priority"], task["submitted_at"], task["extra"])
        )
        conn.commit()

        # 入队
        queue_task = dict(task)
        queue_task["extra"] = extra or {}
        self._pending_queue.put(queue_task)
        self._pending_set.add(task_id)

        log.info("[%s] 任务入队 | priority=%d", task_id[:8], priority)
        return task_id

    def get(self, task_id: str) -> Optional[dict]:
        """
        查询单个任务。返回 dict（含已解析的 extra 字段），
        不存在返回 None。
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, status: Optional[str] = None, search: Optional[str] = None,
             offset: int = 0, limit: int = 50) -> dict:
        """
        任务列表，支持按状态筛选和文本搜索。
        返回: { "tasks": [...], "total": int, "offset": int, "limit": int }
        """
        conn = self._get_conn()
        where_clauses = []
        params = []

        if status:
            where_clauses.append("status=?")
            params.append(status)
        if search:
            where_clauses.append("text LIKE ?")
            params.append(f"%{search}%")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        rows = conn.execute(
            f"SELECT * FROM tasks WHERE {where_sql} "
            f"ORDER BY priority DESC, submitted_at ASC LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()

        total = conn.execute(
            f"SELECT COUNT(*) as c FROM tasks WHERE {where_sql}",
            params
        ).fetchone()["c"]

        return {
            "tasks": [self._row_to_dict(r) for r in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def update_status(self, task_id: str, status: str,
                      file_path: Optional[str] = None,
                      error: Optional[str] = None,
                      started_at: Optional[str] = None,
                      finished_at: Optional[str] = None,
                      **extra_fields):
        """
        更新任务状态和字段。
        - status: 新状态
        - file_path: 生成的文件路径（success 时设置）
        - error: 错误信息（failed 时设置）
        - extra_fields: 其他字段（started_at, finished_at 等）
        """
        updates = ["status=?"]
        params = [status]

        if file_path is not None:
            updates.append("file_path=?")
            params.append(file_path)
        if error is not None:
            updates.append("error=?")
            params.append(error)
        if started_at is not None:
            updates.append("started_at=?")
            params.append(started_at)
        if finished_at is not None:
            updates.append("finished_at=?")
            params.append(finished_at)

        updates.append("finished_at = COALESCE(?, finished_at)")
        params.append(finished_at)

        params.append(task_id)

        conn = self._get_conn()
        conn.execute(
            f"UPDATE tasks SET {', '.join(updates)} WHERE task_id=?",
            params
        )
        conn.commit()
        log.debug("[%s] 状态更新: %s", task_id[:8], status)

    def cancel(self, task_id: str) -> bool:
        """取消 pending/processing 状态的任务。成功返回 True。"""
        task = self.get(task_id)
        if not task:
            return False
        if task["status"] not in (STATUS_PENDING, STATUS_PROCESSING):
            log.warning("[%s] 不可取消，当前状态: %s", task_id[:8], task["status"])
            return False

        conn = self._get_conn()
        conn.execute(
            "UPDATE tasks SET status=?, finished_at=? WHERE task_id=?",
            (STATUS_CANCELLED, self._now(), task_id)
        )
        conn.commit()
        self._pending_set.discard(task_id)
        log.info("[%s] 已取消", task_id[:8])
        return True

    def delete(self, task_id: str, delete_file: bool = True) -> bool:
        """删除任务记录，可选同时删除音频文件。成功返回 True。"""
        task = self.get(task_id)
        if not task:
            return False

        # 删除音频文件
        if delete_file and task.get("file_path"):
            fp = Path(task["file_path"])
            if fp.exists():
                try:
                    fp.unlink()
                    log.debug("[%s] 已删除音频文件: %s", task_id[:8], fp)
                except OSError as e:
                    log.warning("[%s] 删除音频文件失败: %s", task_id[:8], e)

        conn = self._get_conn()
        conn.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
        conn.commit()
        self._pending_set.discard(task_id)
        log.info("[%s] 已删除", task_id[:8])
        return True

    def set_priority(self, task_id: str, priority: int) -> bool:
        """调整任务优先级。成功返回 True。"""
        conn = self._get_conn()
        conn.execute(
            "UPDATE tasks SET priority=? WHERE task_id=?",
            (priority, task_id)
        )
        conn.commit()
        log.info("[%s] 优先级调整为: %d", task_id[:8], priority)
        return True

    def retry(self, task_id: str) -> bool:
        """
        重试失败任务：状态重置为 pending，清除 error/started_at/finished_at。
        同时将任务重新入队。成功返回 True。
        """
        task = self.get(task_id)
        if not task:
            return False
        if task["status"] != STATUS_FAILED:
            log.warning("[%s] 不可重试，当前状态: %s", task_id[:8], task["status"])
            return False

        conn = self._get_conn()
        conn.execute(
            "UPDATE tasks SET status=?, error=NULL, started_at=NULL, "
            "finished_at=NULL WHERE task_id=?",
            (STATUS_PENDING, task_id)
        )
        conn.commit()

        # 重新入队
        task["status"] = STATUS_PENDING
        task["error"] = None
        task["started_at"] = None
        task["finished_at"] = None
        self._pending_queue.put(task)
        self._pending_set.add(task_id)

        log.info("[%s] 已重试", task_id[:8])
        return True

    def stats(self) -> dict:
        """统计概览。"""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) as c FROM tasks").fetchone()["c"]
        by_status_rows = conn.execute(
            "SELECT status, COUNT(*) as c FROM tasks GROUP BY status"
        ).fetchall()

        # 平均处理耗时（秒）
        avg_row = conn.execute(
            "SELECT AVG(julianday(finished_at) - julianday(submitted_at)) * 86400 "
            "as avg_sec FROM tasks WHERE status=? AND finished_at IS NOT NULL "
            "AND started_at IS NOT NULL",
            (STATUS_SUCCESS,)
        ).fetchone()
        avg_sec = round(avg_row["avg_sec"], 1) if avg_row and avg_row["avg_sec"] else 0

        # 队列深度
        queue_depth = conn.execute(
            "SELECT COUNT(*) as c FROM tasks WHERE status=?",
            (STATUS_PENDING,)
        ).fetchone()["c"]

        conn.close()
        self._local.conn = None

        return {
            "total": total,
            "by_status": {r["status"]: r["c"] for r in by_status_rows},
            "avg_processing_seconds": avg_sec,
            "queue_depth": queue_depth,
        }

    def cleanup(self, days: int = 30, status: Optional[str] = None,
                delete_file: bool = True) -> int:
        """
        批量清理旧任务。
        - days: 保留天数（超过此天数的任务被清理）
        - status: 可选，只清理特定状态的任务
        - delete_file: 是否同时删除音频文件
        返回删除的任务数量。
        """
        conn = self._get_conn()
        where = ["submitted_at < datetime('now', ?)"]
        params = [f"-{days} days"]

        if status:
            where.append("status=?")
            params.append(status)

        rows = conn.execute(
            f"SELECT * FROM tasks WHERE {' AND '.join(where)}", params
        ).fetchall()

        deleted = 0
        for row in rows:
            task = self._row_to_dict(row)
            tid = task["task_id"]
            if delete_file and task.get("file_path"):
                fp = Path(task["file_path"])
                if fp.exists():
                    try:
                        fp.unlink()
                    except OSError:
                        pass
            conn.execute("DELETE FROM tasks WHERE task_id=?", (tid,))
            self._pending_set.discard(tid)
            deleted += 1

        conn.commit()
        log.info("清理完成: 删除 %d 条任务 (days=%d, status=%s, delete_file=%s)",
                 deleted, days, status or "all", delete_file)
        return deleted

    # ── 队列操作（供 worker 使用）─────────────────────────

    def get_pending_task(self, block: bool = True, timeout: Optional[float] = None) -> Optional[dict]:
        """
        从 pending 队列取一个任务（供 worker 调用）。
        返回 task dict，如果没有 pending 任务则阻塞等待。
        """
        try:
            task = self._pending_queue.get(block=block, timeout=timeout)
            self._pending_set.discard(task["task_id"])
            return task
        except Exception:
            return None

    def mark_task_done(self):
        """标记队列任务完成。"""
        self._pending_queue.task_done()

    def qsize(self) -> int:
        """当前 pending 队列大小。"""
        return self._pending_queue.qsize()