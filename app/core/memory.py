"""会话记忆：支撑多轮对话。

刻意保持「进程内存 + 容量上限」的实现：
- 无需额外存储依赖，进程重启后会话清除
- 生产环境替换为 Redis / 数据库只需实现同一套方法（接口稳定，调用方无感）
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import settings


class ConversationMemory:
    """按 ``session_id`` 保存最近若干轮对话。线程安全，带会话数量上限。"""

    def __init__(self, max_turns: int | None = None, max_sessions: int = 1000) -> None:
        self._max_turns = max_turns or settings.max_history_turns
        self._max_sessions = max_sessions
        self._store: OrderedDict[str, deque[BaseMessage]] = OrderedDict()
        self._lock = threading.Lock()

    def _bucket(self, session_id: str) -> deque[BaseMessage]:
        bucket = self._store.get(session_id)
        if bucket is None:
            bucket = deque(maxlen=self._max_turns * 2)  # 1 轮 = 用户 + 助手
            self._store[session_id] = bucket
            # LRU：超量时淘汰最久未使用的会话，避免内存无限增长
            while len(self._store) > self._max_sessions:
                self._store.popitem(last=False)
        else:
            self._store.move_to_end(session_id)
        return bucket

    def add_user(self, session_id: str, content: str) -> None:
        with self._lock:
            self._bucket(session_id).append(HumanMessage(content=content))

    def add_assistant(self, session_id: str, content: str) -> None:
        with self._lock:
            self._bucket(session_id).append(AIMessage(content=content))

    def history(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._store.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._store)
