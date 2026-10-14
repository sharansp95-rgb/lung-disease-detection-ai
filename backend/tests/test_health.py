import main

ALL_MODELS = {"xray", "ct", "gate", "xray_sanity", "ct_sanity"}


# Models lazy-load on the first prediction request (main._ensure_models_loaded),
# not at startup, so what / and /health report depends on whether a prediction
# has already happened in this test session. These tests check both states
# rather than assuming an order.


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == ("ok" if main._models_loaded else "degraded")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == (ALL_MODELS if main._models_loaded else set())
    assert "google_oauth_configured" in body
    assert "email_configured" in body


def test_first_prediction_lazy_loads_all_models(client, xray_sample_path):
    with open(xray_sample_path, "rb") as f:
        resp = client.post("/predict/xray", files={"file": ("sample.png", f, "image/png")})
    assert resp.status_code == 200, resp.text
    assert main._models_loaded

    health = client.get("/health").json()
    assert set(health["models_loaded"]) == ALL_MODELS
    assert client.get("/").json()["status"] == "ok"
