import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


class TestRoutes:
    def test_openapi_paths(self, client):
        assert set(client.get("/openapi.json").json()["paths"]) == {"/api/v1/agent/chat"}

    def test_docs(self, client):
        assert client.get("/docs").status_code == 200

    @pytest.mark.parametrize("path", ["/health", "/api/v1/jev/status"])
    def test_removed_get_routes(self, client, path):
        assert client.get(path).status_code == 404

    def test_removed_analyze_route(self, client):
        assert client.post("/api/v1/jev/analyze", json={"question": "q"}).status_code == 404

    def test_agent_is_created_on_startup(self, client):
        assert client.app.state.agent is not None

    def test_openapi_documents_camel_case(self, client):
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        assert "threadId" in schemas["ChatRequest"]["properties"]
        assert "contextProbability" in schemas["ChatResponse"]["properties"]


class TestPlayground:
    @pytest.fixture
    def html(self):
        return (ROOT / "app/playground/index.html").read_text(encoding="utf-8")

    def test_served_as_html(self, client):
        response = client.get("/playground")
        assert response.status_code == 200 and "text/html" in response.headers["content-type"]

    def test_served_from_any_working_directory(self, client, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert client.get("/playground").status_code == 200

    def test_hidden_from_openapi(self, client):
        assert "/playground" not in client.get("/openapi.json").json()["paths"]

    def test_calls_existing_route_only(self, html):
        assert "/api/v1/agent/chat" in html and "/api/v1/jev" not in html

    def test_does_not_build_html_from_responses(self, html):
        assert "innerHTML" not in html and "document.write" not in html

    def test_no_external_resources(self, html):
        assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html

    def test_has_dark_mode_and_viewport(self, html):
        assert "prefers-color-scheme: dark" in html and 'name="viewport"' in html


class TestProjectFiles:
    def test_real_env_is_ignored(self):
        assert ".env" in (ROOT / ".gitignore").read_text().splitlines()

    def test_env_example_has_no_values_for_secrets(self):
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if line.endswith("=") or line.startswith("#") or not line:
                continue
            name = line.split("=", 1)[0]
            assert not name.endswith("_API_KEY") and name != "DATABASE_URL", name

    def test_env_example_lists_every_setting(self):
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        config = (ROOT / "app/core/config.py").read_text(encoding="utf-8")
        for name in re.findall(r'getenv\("([A-Z_]+)"\)', config):
            assert f"{name}=" in example, name

    def test_no_secret_patterns_in_source(self):
        pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{20,}|AQ\.[A-Za-z0-9_-]{20,}")
        files = [*ROOT.joinpath("app").rglob("*"), ROOT / "main.py", ROOT / ".env.example", ROOT / "requirements.txt"]
        leaked = [str(f) for f in files if f.is_file() and f.suffix != ".pyc" and pattern.search(f.read_text("utf-8"))]
        assert leaked == []

    def test_requirements_are_pinned(self):
        for line in (ROOT / "requirements.txt").read_text().splitlines():
            assert "==" in line, line

    def test_direct_imports_are_in_requirements(self):
        requirements = (ROOT / "requirements.txt").read_text().lower()
        packages = [
            "fastapi",
            "uvicorn",
            "langgraph",
            "langchain-core",
            "langchain==",
            "langchain-text-splitters",
            "psycopg",
            "neo4j",
            "qdrant-client",
            "langchain-aws",
            "typesafe-sdk",
            "python-dotenv",
            "pydantic",
        ]
        for package in packages:
            assert package in requirements, package
