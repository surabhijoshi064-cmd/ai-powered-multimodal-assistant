def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "dependencies" in data
    assert data["dependencies"]["database"] == "connected"
    assert data["dependencies"]["storage"] == "accessible"
    assert "configured_providers" in data["dependencies"]
    # Ensure no secrets or API keys are leaked
    for key, value in data["dependencies"]["configured_providers"].items():
        assert "key" not in key.lower()
        assert not value.startswith("sk-")
