import time
from collections import OrderedDict
from typing import Callable

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

from app.retrieval.schemas import Chunk

MAX_CONVERSATIONS = 1000
TTL_SECONDS = 3600
MAX_MESSAGES = 6


class Conversation(TypedDict):
    messages: list[AnyMessage]
    context: list[Chunk]


class ConversationStore:
    """대화당 최근 메시지와 컨텍스트만 메모리에 보관한다. 개수와 시간에 상한이 있고 재시작하면 사라진다."""

    def __init__(
        self,
        max_conversations: int = MAX_CONVERSATIONS,
        ttl_seconds: float = TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_conversations = max_conversations
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, Conversation]] = OrderedDict()

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
