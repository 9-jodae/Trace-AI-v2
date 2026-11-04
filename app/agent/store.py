import asyncio
import time
from collections import OrderedDict
from typing import Callable
from weakref import WeakValueDictionary

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

from app.core import config
from app.retrieval.schemas import Chunk

MAX_MESSAGES = 6


class Conversation(TypedDict):
    messages: list[AnyMessage]
    context: list[Chunk]


class ConversationStore:
    """대화당 최근 메시지와 컨텍스트만 메모리에 보관한다. 개수와 시간에 상한이 있고 재시작하면 사라진다."""

    def __init__(
        self,
        max_conversations: int = config.DEFAULT_MAX_CONVERSATIONS,
        ttl_seconds: float = config.DEFAULT_CONVERSATION_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_conversations = max_conversations
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, Conversation]] = OrderedDict()
        # 쓰는 요청이 있는 동안만 잠금이 살아 있어 대화 수만큼 쌓이지 않는다
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock
        return lock

    def __len__(self) -> int:
        return len(self._items)

    def _evict(self) -> None:
        deadline = self._clock() - self._ttl_seconds
        while self._items:
            oldest_id, (saved_at, _) = next(iter(self._items.items()))
            if saved_at > deadline and len(self._items) <= self._max_conversations:
                break
            del self._items[oldest_id]

    def get(self, thread_id: str) -> Conversation:
        self._evict()
        item = self._items.get(thread_id)
        return item[1] if item else {"messages": [], "context": []}

    def save(self, thread_id: str, messages: list[AnyMessage], context: list[Chunk]) -> None:
        self._items.pop(thread_id, None)
        self._items[thread_id] = (self._clock(), {"messages": messages[-MAX_MESSAGES:], "context": context})
        self._evict()
