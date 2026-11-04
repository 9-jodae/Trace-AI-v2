"""#7 완료 조건 중 메모리 상한."""

import pickle

from conftest import chunk
from langchain_core.messages import HumanMessage

from app.agent.store import MAX_MESSAGES, ConversationStore


def message(text="q"):
    return [HumanMessage(content=text)]


def test_unknown_thread_is_empty():
    assert ConversationStore().get("none") == {"messages": [], "context": []}


def test_save_and_get():
    store = ConversationStore()
    store.save("t", message("q"), [chunk("a")])
    saved = store.get("t")
    assert saved["messages"][0].text == "q" and saved["context"] == [chunk("a")]


def test_messages_are_trimmed_to_recent():
    store = ConversationStore()
    store.save("t", [HumanMessage(content=str(i)) for i in range(50)], [])
    texts = [m.text for m in store.get("t")["messages"]]
    assert len(texts) == MAX_MESSAGES and texts[-1] == "49"


def test_oldest_conversation_is_evicted():
    store = ConversationStore(max_conversations=2)
    store.save("a", message(), [])
    store.save("b", message(), [])
    store.save("c", message(), [])
    assert len(store) == 2 and store.get("a")["messages"] == [] and store.get("c")["messages"]


def test_saving_again_refreshes_order():
    store = ConversationStore(max_conversations=2)
    store.save("a", message(), [])
    store.save("b", message(), [])
    store.save("a", message(), [])
    store.save("c", message(), [])
    assert store.get("a")["messages"] and store.get("b")["messages"] == []


def test_expired_conversation_is_gone():
    now = [0.0]
    store = ConversationStore(ttl_seconds=10, clock=lambda: now[0])
    store.save("a", message(), [chunk("a")])
    now[0] = 9.9
    assert store.get("a")["context"]
    now[0] = 10.0
    assert store.get("a") == {"messages": [], "context": []} and len(store) == 0


def test_size_does_not_grow_with_turns(ask, client):
    def size():
        return len(pickle.dumps(client.app.state.conversations._items))

    ask("q")
    after_one = size()
    ask("q")
    ask("q")
    ask("q")
    after_few = size()
    for _ in range(200):
        ask("q")
    assert size() <= after_few + 64
    assert after_one > 0


def test_many_threads_stay_within_limit(client, ai, monkeypatch):
    store = client.app.state.conversations
    monkeypatch.setattr(store, "_max_conversations", 5)
    for index in range(20):
        client.post("/api/v1/agent/chat", json={"message": "q", "threadId": f"t{index}"})
    assert len(store) == 5
