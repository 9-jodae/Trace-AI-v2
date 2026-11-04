"""#5, #6 완료 조건: 운영 환경의 인증, 문서 차단, 샘플 데이터 금지."""

import pytest
from fastapi.testclient import TestClient

import main
from app.core import config
from app.core.errors import AiUnavailableError
from app.retrieval import service

URL = "/api/v1/agent/chat"
SECRET = "운영-비밀".encode().hex()


@pytest.fixture
def production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("INTERNAL_SECRET", SECRET)
    with TestClient(main.create_app(), raise_server_exceptions=False) as client:
        yield client


class TestStartup:
    def test_production_is_the_default(self, monkeypatch):
        monkeypatch.delenv("APP_ENV")
        monkeypatch.setenv("INTERNAL_SECRET", "")
        with pytest.raises(RuntimeError, match="INTERNAL_SECRET"):
            main.create_app()

    @pytest.mark.parametrize("value", ["prod", "LOCAL", "staging", ""])
    def test_anything_but_local_is_production(self, monkeypatch, value):
        monkeypatch.setenv("APP_ENV", value)
        monkeypatch.setenv("INTERNAL_SECRET", "")
        with pytest.raises(RuntimeError):
            main.create_app()

    def test_production_refuses_sample_data(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("USE_SAMPLE_DATA", "true")
        with pytest.raises(RuntimeError, match="USE_SAMPLE_DATA"):
            main.create_app()


class TestInternalSecret:
    @pytest.mark.parametrize("headers", [{}, {"X-Internal-Secret": ""}, {"X-Internal-Secret": "wrong"}])
    def test_rejects_missing_or_wrong_secret(self, production, ai, headers):
        response = production.post(URL, json={"message": "q"}, headers=headers)
        assert response.status_code == 401
        assert response.json()["code"] == "UNAUTHORIZED"
        assert ai.search_calls == []

    def test_rejects_before_validating_body(self, production):
        assert production.post(URL, json={}).status_code == 401

    def test_non_ascii_secret_does_not_crash(self, production):
        response = production.post(URL, json={"message": "q"}, headers={"X-Internal-Secret": "한글".encode()})
        assert response.status_code == 401

    def test_accepts_correct_secret(self, production, ai):
        response = production.post(URL, json={"message": "q"}, headers={"X-Internal-Secret": SECRET})
        assert response.status_code == 200

    def test_secret_is_not_echoed(self, production):
        response = production.post(URL, json={"message": "q"}, headers={"X-Internal-Secret": "wrong"})
        assert "wrong" not in response.text and SECRET not in response.text

    def test_local_needs_no_secret(self, client, ai):
        assert client.post(URL, json={"message": "q"}).status_code == 200


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/playground"])
def test_production_closes_developer_pages(production, path):
    response = production.get(path)
    assert response.status_code == 404 and response.json()["code"] == "NOT_FOUND"


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/playground"])
def test_local_keeps_developer_pages(client, path):
    assert client.get(path).status_code == 200


class TestSampleDataNeverReachesProduction:
    def post(self, client):
        return client.post(URL, json={"message": "재택근무 규정 알려줘"}, headers={"X-Internal-Secret": SECRET})

    def test_production_returns_503_not_samples(self, production, monkeypatch):
        response = self.post(production)
        assert response.status_code == 503
        assert "remote" not in response.text and "payment" not in response.text

    def test_setting_cannot_be_turned_on_after_startup(self, production, monkeypatch):
        monkeypatch.setenv("USE_SAMPLE_DATA", "true")
        # 응답 코드만 보면 LLM 실패로도 503이 되므로 저장소 선택을 직접 확인한다
        assert config.use_sample_data() is False
        with pytest.raises(AiUnavailableError):
            service.get_stores()
        assert self.post(production).status_code == 503

    def test_local_without_opt_in_returns_503(self, client):
        response = client.post(URL, json={"message": "재택근무 규정 알려줘"})
        assert response.status_code == 503
