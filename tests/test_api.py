from fastapi.testclient import TestClient

from main import app


def test_empty_message_returns_invalid_request():
    response = TestClient(app).post("/api/v1/agent/chat", json={"message": ""})

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"
