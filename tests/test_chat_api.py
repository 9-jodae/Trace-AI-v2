import pytest
from conftest import chunk

from app.agent import graph
from app.agent.schemas import MAX_MESSAGE_LENGTH

URL = "/api/v1/agent/chat"
RESPONSE_FIELDS = {
    "threadId",
    "answer",
    "route",
    "contextProbability",
    "sources",
    "evidence",
    "failedSources",
    "fallbackReason",
}


def test_response_fields_are_camel_case(ask):
    assert set(ask("q").json()) == RESPONSE_FIELDS


def test_source_fields(ask):
    assert set(ask("q").json()["sources"][0]) == {
        "id",
        "text",
        "source",
        "documentId",
        "versionId",
        "charStart",
        "charEnd",
    }


def test_thread_id_is_returned(ask):
    assert ask("q", "my-thread").json()["threadId"] == "my-thread"


@pytest.mark.parametrize(
    "payload",
    [{"message": "q"}, {"message": "q", "threadId": None}, {"message": "q", "threadId": ""}],
)
def test_thread_id_is_generated(client, ai, payload):
    assert len(client.post(URL, json=payload).json()["threadId"]) == 36


def test_generated_thread_ids_differ(client, ai):
    first = client.post(URL, json={"message": "q"}).json()["threadId"]
    assert first != client.post(URL, json={"message": "q"}).json()["threadId"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"message": ""}, {"message": None}, {"message": 123}, {"message": "q", "threadId": 1}],
)
def test_invalid_payload(client, ai, payload):
    response = client.post(URL, json=payload)
    assert response.status_code == 422 and response.json()["code"] == "INVALID_REQUEST"
    assert ai.search_calls == []


@pytest.mark.parametrize("message", ["   ", "\n\t"])
def test_blank_message_does_not_spend_api(client, ai, message):
    assert client.post(URL, json={"message": message}).status_code == 422
    assert ai.search_calls == []


def test_oversized_message_is_rejected(client, ai):
    assert client.post(URL, json={"message": "가" * (MAX_MESSAGE_LENGTH + 1)}).status_code == 422
    assert ai.search_calls == []


def test_malformed_json(client):
    response = client.post(URL, content="{bad", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_get_is_not_allowed(client):
    assert client.get(URL).status_code == 405


def test_unexpected_error_is_500(client, ai):
    ai.search_result = RuntimeError("비밀")
    response = client.post(URL, json={"message": "q"})
    assert response.status_code == 500 and response.json()["code"] == "INTERNAL_ERROR"
    assert "비밀" not in response.text


def test_special_characters_round_trip(ask):
    question = "E4012 <script>alert(1)</script> 'quote' \"double\" 😀"
    assert ask(question).json()["answer"] == f"답변: {question}"


def test_null_source_from_database_does_not_crash(ask, ai):
    ai.search_result = {"chunks": [{"id": "a", "text": "내용", "source": None}], "failed_sources": []}
    assert ask("q").status_code == 200


def test_max_length_message_is_accepted(client, ai):
    assert client.post(URL, json={"message": "가" * MAX_MESSAGE_LENGTH}).status_code == 200


def test_surrounding_whitespace_is_trimmed(ask, ai):
    ask("  질문  ")
    assert ai.search_calls == ["질문"]


def test_contract_fields_reach_response(ask, ai):
    full = {
        "id": "u",
        "text": "t",
        "source": "s",
        "document_id": "d",
        "version_id": "v",
        "char_start": 3,
        "char_end": 9,
    }
    ai.search_result = {"chunks": [full], "failed_sources": []}
    source = ask("q").json()["sources"][0]
    assert source["documentId"] == "d" and source["versionId"] == "v"
    assert source["charStart"] == 3 and source["charEnd"] == 9


def test_oversized_thread_id_is_rejected(client, ai):
    assert client.post(URL, json={"message": "q", "threadId": "x" * 129}).status_code == 422


class TestEvidence:
    @pytest.fixture
    def grounded(self, ai, monkeypatch):
        ai.search_result = {"chunks": [chunk("a", "연차는 15일입니다.")], "failed_sources": []}

        async def generate(question, context, history):
            ai.llm_calls.append({"history": [message.text for message in history]})
            quotes = '[{"chunk": 1, "quote": "15일"}, {"chunk": 1, "quote": "20일"}]'
            return f'{{"answer": "15일입니다.", "evidence": {quotes}}}'

        monkeypatch.setattr(graph, "generate_answer", generate)
        return ai

    def test_fields_and_only_grounded_quotes_are_returned(self, ask, grounded):
        body = ask("q").json()
        assert body["answer"] == "15일입니다."
        assert [item["quote"] for item in body["evidence"]] == ["15일"]
        assert set(body["evidence"][0]) == {
            "chunkId",
            "source",
            "documentId",
            "versionId",
            "quote",
            "charStart",
            "charEnd",
        }

    def test_history_keeps_answer_not_json(self, ask, grounded):
        ask("q1")
        ask("q2")
        assert grounded.llm_calls[-1]["history"][-1] == "15일입니다."

    def test_broken_json_is_unavailable_and_not_saved(self, ask, ai, monkeypatch):
        async def generate(question, context, history):
            return '{"answer": "잘린'

        monkeypatch.setattr(graph, "generate_answer", generate)
        response = ask("q")
        assert response.status_code == 503 and "잘린" not in response.text
        monkeypatch.setattr(graph, "generate_answer", ai.generate)
        ask("다음 질문")
        assert ai.llm_calls[-1]["history"] == []

    def test_no_result_has_no_evidence(self, ask, ai):
        ai.search_result = {"chunks": [], "failed_sources": []}
        assert ask("q").json()["evidence"] == []


class TestFilters:
    BODY = {
        "extensions": ["pdf", "md"],
        "documentTypes": ["REPORT"],
        "sourceKinds": ["LOCAL"],
        "authoredFrom": "2026-01-01",
        "authoredTo": "2026-12-31",
    }

    def test_camel_case_filters_reach_search(self, client, ai):
        response = client.post(URL, json={"message": "q", "filters": self.BODY})
        assert response.status_code == 200
        filters = ai.search_filters
        assert filters.extensions == ["pdf", "md"] and filters.document_types == ["REPORT"]
        assert filters.source_kinds == ["LOCAL"] and filters.authored_to.isoformat() == "2026-12-31"

    def test_no_filters_is_none(self, client, ai):
        client.post(URL, json={"message": "q"})
        assert ai.search_filters is None

    def test_null_field_turns_filter_off(self, client, ai):
        client.post(URL, json={"message": "q", "filters": {"extensions": None, "sourceKinds": ["LOCAL"]}})
        assert ai.search_filters.extensions is None and ai.search_filters.source_kinds == ["LOCAL"]

    @pytest.mark.parametrize("filters", [{"extensions": []}, {"authoredFrom": "어제"}, {"sourceKinds": "LOCAL"}])
    def test_invalid_filters_are_rejected(self, client, ai, filters):
        assert client.post(URL, json={"message": "q", "filters": filters}).status_code == 422
        assert ai.search_calls == []

    def test_filtered_followup_searches_again(self, client, ai):
        client.post(URL, json={"message": "q1", "threadId": "t"})
        client.post(URL, json={"message": "q2", "threadId": "t", "filters": {"extensions": ["pdf"]}})
        assert len(ai.search_calls) == 2 and len(ai.jev_calls) == 0


class TestAccessScope:
    SCOPE = {
        "departmentId": "11111111-1111-1111-1111-111111111111",
        "position": "STAFF",
        "clearance": "INTERNAL",
        "isAdmin": False,
    }

    def test_scope_reaches_search(self, client, ai):
        assert client.post(URL, json={"message": "q", "accessScope": self.SCOPE}).status_code == 200
        assert ai.search_scope == ("11111111-1111-1111-1111-111111111111", "STAFF", "INTERNAL", False)

    def test_same_scope_is_used_for_context_revalidation(self, client, ai):
        client.post(URL, json={"message": "q1", "threadId": "t", "accessScope": self.SCOPE})
        client.post(URL, json={"message": "q2", "threadId": "t", "accessScope": {**self.SCOPE, "isAdmin": True}})
        assert ai.filter_scope[3] is True

    def test_missing_scope_is_none(self, client, ai):
        client.post(URL, json={"message": "q"})
        assert ai.search_scope is None

    @pytest.mark.parametrize(
        "change",
        [
            {"departmentId": "not-uuid"},
            {"departmentId": None},
            {"position": None},
            {"clearance": None},
            {"isAdmin": "true"},
            {"isAdmin": None},
            {"position": "x" * 31},
        ],
    )
    def test_invalid_scope_is_rejected_before_search(self, client, ai, change):
        body = {"message": "q", "accessScope": {**self.SCOPE, **change}}
        assert client.post(URL, json=body).status_code == 422
        assert ai.search_calls == []

    def test_incomplete_scope_is_rejected(self, client, ai):
        assert client.post(URL, json={"message": "q", "accessScope": {"isAdmin": True}}).status_code == 422
        assert ai.search_calls == []
