from fastapi.testclient import TestClient

from rag_anchor.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_health_returns_expected_json() -> None:
    response = client.get("/api/v1/health")

    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "ok", "version": "0.1.0"}
