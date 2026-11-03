from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    assert client.get("/").status_code == 200


def test_health():
    assert client.get("/health").status_code == 200


def test_heartbeat():
    assert client.get("/heartbeat").status_code == 200


def test_tools():
    assert client.get("/tools").status_code == 200


def test_current_moon():
    response = client.get("/v1/moon/current")
    assert response.status_code == 200
    data = response.json()
    assert "phase" in data or "illumination_percent" in data


def test_moon_phases():
    response = client.get("/v1/moon/phases")
    assert response.status_code == 200


def test_hijri_info():
    response = client.get("/v1/moon/hijri")
    assert response.status_code == 200


def test_moon_forecast():
    response = client.get("/v1/moon/forecast?days=7")
    assert response.status_code == 200


def test_mcp_get_lunar_status():
    response = client.post(
        "/mcp/get_lunar_status",
        json={"lat": 3.139, "lon": 101.6869, "timezone": "Asia/Kuala_Lumpur"},
    )
    assert response.status_code == 200
