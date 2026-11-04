"""#7 완료 조건 중 메모리 상한."""

import asyncio
import gc
import pickle

import httpx
import pytest
from conftest import chunk
from langchain_core.messages import HumanMessage

import main
from app.agent import graph
from app.agent.store import MAX_MESSAGES, ConversationStore
from app.core.errors import AiUnavailableError


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


class TestConcurrentRequests:
    @pytest.fixture
    def app(self, ai):
        application = main.create_app()
        application.state.agent = graph.build_agent()
        application.state.conversations = ConversationStore()
        return application

    @staticmethod
    async def send(app, *requests):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await asyncio.gather(
                *(
                    client.post("/api/v1/agent/chat", json={"message": message, "threadId": thread})
                    for message, thread in requests
                )
            )

    def test_same_thread_keeps_every_turn(self, app, ai):
        async def slow_generate(question, context, history):
            ai.llm_calls.append(question)
            await asyncio.sleep(0.01)
            return f"답변: {question}"

        graph.generate_answer = slow_generate
        responses = asyncio.run(self.send(app, ("q1", "t"), ("q2", "t"), ("q3", "t")))
        assert all(response.status_code == 200 for response in responses)
        texts = [message.text for message in app.state.conversations.get("t")["messages"]]
        assert [text for text in texts if text.startswith("q")] == ["q1", "q2", "q3"]

    def test_same_thread_runs_one_at_a_time(self, app, ai):
        running = {"now": 0, "max": 0}

        async def slow_generate(question, context, history):
            running["now"] += 1
            running["max"] = max(running["max"], running["now"])
            await asyncio.sleep(0.01)
            running["now"] -= 1
            return "답변"

        graph.generate_answer = slow_generate
        asyncio.run(self.send(app, ("q1", "t"), ("q2", "t")))
        assert running["max"] == 1

    def test_different_threads_run_together(self, app, ai):
        running = {"now": 0, "max": 0}

        async def slow_generate(question, context, history):
            running["now"] += 1
            running["max"] = max(running["max"], running["now"])
            await asyncio.sleep(0.01)
            running["now"] -= 1
            return "답변"

        graph.generate_answer = slow_generate
        asyncio.run(self.send(app, ("q1", "a"), ("q2", "b")))
        assert running["max"] == 2

    def test_failure_releases_the_lock(self, app, ai):
        ai.llm_error = AiUnavailableError("x")
        assert asyncio.run(self.send(app, ("q1", "t")))[0].status_code == 503
        ai.llm_error = None
        assert asyncio.run(self.send(app, ("q2", "t")))[0].status_code == 200

    def test_locks_do_not_accumulate(self, app, ai):
        asyncio.run(self.send(app, *[("q", f"t{index}") for index in range(20)]))
        gc.collect()
        assert len(app.state.conversations._locks) == 0

