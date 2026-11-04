import pytest

URL = "/api/v1/agent/chat"
RESPONSE_FIELDS = {
    "threadId",
    "answer",
    "route",
    "contextProbability",
    "sources",
    "failedSources",
    "fallbackReason",
}


def test_response_fields_are_camel_case(ask):
    assert set(ask("q").json()) == RESPONSE_FIELDS


def test_source_fields(ask):
    assert set(ask("q").json()["sources"][0]) == {"id", "text", "source"}


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


@pytest.mark.xfail(strict=True, reason="결함 5: 공백뿐인 질문도 검색과 LLM까지 실행됨")
@pytest.mark.parametrize("message", ["   ", "\n\t"])
def test_blank_message_does_not_spend_api(client, ai, message):
    assert client.post(URL, json={"message": message}).status_code == 422
    assert ai.search_calls == []


@pytest.mark.xfail(strict=True, reason="결함 6: 질문 길이 제한이 없어 비용이 무제한으로 늘 수 있음")
def test_oversized_message_is_rejected(client, ai):
    assert client.post(URL, json={"message": "가" * 1_000_000}).status_code == 422
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


@pytest.mark.xfail(strict=True, reason="결함 4: source가 NULL이면 LLM 비용을 쓴 뒤 500 발생")
def test_null_source_from_database_does_not_crash(ask, ai):
    ai.search_result = {"chunks": [{"id": "a", "text": "내용", "source": None}], "failed_sources": []}
    assert ask("q").status_code == 200
