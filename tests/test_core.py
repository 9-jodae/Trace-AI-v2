import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import Field

from app.core import config
from app.core.errors import AiUnavailableError, register_error_handlers
from app.core.schemas import CamelModel, ErrorResponse


class TestConfig:
    def test_reads_values(self):
        assert config.jev_model() == "jev-test"
        assert config.embedding_model() == "openai:fake-embedding"
        assert config.llm_model() == "fake:fake-llm"

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_means_unset(self, monkeypatch, value):
        for name in ["JEV_API_KEY", "EMBEDDING_API_KEY", "EMBEDDING_MODEL", "LLM_API_KEY", "LLM_MODEL", "DATABASE_URL"]:
            monkeypatch.setenv(name, "") if value == "" else monkeypatch.delenv(name)
        assert config.jev_api_key() is None
        assert config.embedding_api_key() is None
        assert config.embedding_model() is None
        assert config.llm_api_key() is None
        assert config.llm_model() is None
        assert config.database_url() is None

    def test_jev_model_default(self, monkeypatch):
        monkeypatch.setenv("JEV_MODEL", "")
        assert config.jev_model() == "jev-latest"

    def test_embedding_dimensions_default_is_titan_1024(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "")
        assert config.embedding_dimensions() == 1024

    def test_embedding_dimensions_is_int(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
        assert config.embedding_dimensions() == 768

    def test_embedding_dimensions_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "abc")
        with pytest.raises(ValueError):
            config.embedding_dimensions()

    def test_vector_distance_cutoff_default(self, monkeypatch):
        monkeypatch.delenv("VECTOR_MAX_COSINE_DISTANCE", raising=False)
        assert config.vector_max_cosine_distance() == 0.55

    def test_vector_distance_cutoff_can_be_configured(self, monkeypatch):
        monkeypatch.setenv("VECTOR_MAX_COSINE_DISTANCE", "0.42")
        assert config.vector_max_cosine_distance() == 0.42

    @pytest.mark.parametrize("value", ["-0.1", "2.1", "invalid"])
    def test_vector_distance_cutoff_rejects_invalid_values(self, monkeypatch, value):
        monkeypatch.setenv("VECTOR_MAX_COSINE_DISTANCE", value)
        with pytest.raises(ValueError):
            config.vector_max_cosine_distance()


class TestServerOptions:
    @pytest.fixture(autouse=True)
    def clean(self, monkeypatch):
        monkeypatch.delenv("HOST", raising=False)
        monkeypatch.delenv("PORT", raising=False)

    def test_local_binds_loopback_with_reload(self):
        assert config.server_options() == {"host": "127.0.0.1", "port": 8000, "reload": True}

    def test_production_has_no_reload(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        assert config.server_options() == {"host": "0.0.0.0", "port": 8000, "reload": False}

    def test_missing_app_env_has_no_reload(self, monkeypatch):
        monkeypatch.delenv("APP_ENV")
        assert config.server_options()["reload"] is False

    def test_host_and_port_can_be_set(self, monkeypatch):
        monkeypatch.setenv("HOST", "10.0.0.1")
        monkeypatch.setenv("PORT", "9000")
        options = config.server_options()
        assert options["host"] == "10.0.0.1" and options["port"] == 9000


class TestCamelModel:
    class Sample(CamelModel):
        thread_id: str
        fallback_reason: str | None = None

    def test_accepts_camel_and_snake(self):
        assert self.Sample(threadId="a").thread_id == "a"
        assert self.Sample(thread_id="a").thread_id == "a"

    def test_dumps_camel(self):
        assert self.Sample(thread_id="a").model_dump(by_alias=True) == {"threadId": "a", "fallbackReason": None}

    def test_error_response(self):
        assert ErrorResponse(code="C", message="M").model_dump(by_alias=True) == {"code": "C", "message": "M"}


class TestErrorHandlers:
    @pytest.fixture
    def error_client(self):
        app = FastAPI()
        register_error_handlers(app)

        class Body(CamelModel):
            value: str = Field(min_length=1)

        @app.get("/unavailable")
        async def unavailable():
            raise AiUnavailableError("비밀 내부 정보")

        @app.get("/boom")
        async def boom():
            raise RuntimeError("비밀 내부 정보")

        @app.post("/body")
        async def body(payload: Body):
            return payload

        return TestClient(app, raise_server_exceptions=False)

    def test_ai_unavailable_is_503(self, error_client):
        response = error_client.get("/unavailable")
        assert response.status_code == 503
        assert response.json() == {"code": "AI_UNAVAILABLE", "message": "현재 답변을 생성할 수 없습니다."}

    def test_unexpected_is_500(self, error_client):
        response = error_client.get("/boom")
        assert response.status_code == 500
        assert response.json()["code"] == "INTERNAL_ERROR"

    @pytest.mark.parametrize("path", ["/unavailable", "/boom"])
    def test_internal_detail_is_hidden(self, error_client, path):
        assert "비밀" not in error_client.get(path).text

    def test_validation_is_422_with_field(self, error_client):
        response = error_client.post("/body", json={"value": ""})
        assert response.status_code == 422
        assert response.json()["code"] == "INVALID_REQUEST"
        assert "body.value" in response.json()["message"]

    def test_malformed_json_is_422(self, error_client):
        response = error_client.post("/body", content="{bad", headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["code"] == "INVALID_REQUEST"

    def test_not_found_uses_same_format(self, error_client):
        response = error_client.get("/none")
        assert response.status_code == 404
        assert set(response.json()) == {"code", "message"}

    def test_method_not_allowed_uses_same_format(self, error_client):
        response = error_client.delete("/boom")
        assert response.status_code == 405
        assert set(response.json()) == {"code", "message"}
