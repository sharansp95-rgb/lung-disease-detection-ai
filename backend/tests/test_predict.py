"""
Exercises the real prediction pipeline end to end (modality gate -> OOD ->
classifier -> Grad-CAM -> report) against real sample images from
data/{xray,ct_scan}/test -- no mocking. Slower than a unit test, but this
is the pipeline the app actually depends on; the startup reproduction
checks (backend/models.py) verify the *models* are correct, this verifies
the *route* wires them together correctly.
"""

import io

from PIL import Image

XRAY_CLASSES = {"COVID-19", "Normal", "Pneumonia", "Tuberculosis"}
CT_CLASSES = {"Benign", "Malignant", "Normal"}


def test_predict_xray_with_real_sample(client, xray_sample_path):
    with open(xray_sample_path, "rb") as f:
        resp = client.post("/predict/xray", files={"file": ("sample.png", f, "image/png")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["modality"] == "xray"
    assert body["predicted_class"] in XRAY_CLASSES
    assert set(body["class_probabilities"].keys()) == XRAY_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["heatmap_png_base64"].startswith("data:image")
    assert body["report_source"] in ("groq", "template_fallback")


def test_predict_ct_with_real_sample(client, ct_sample_path):
    with open(ct_sample_path, "rb") as f:
        resp = client.post("/predict/ct", files={"file": ("sample.jpg", f, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["modality"] == "ct"
    assert body["predicted_class"] in CT_CLASSES
    assert set(body["class_probabilities"].keys()) == CT_CLASSES


def test_predict_saves_scan_history_when_authenticated(client, xray_sample_path, verified_user):
    token, _, _ = verified_user
    with open(xray_sample_path, "rb") as f:
        resp = client.post(
            "/predict/xray",
            files={"file": ("sample.png", f, "image/png")},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200

    history = client.get("/scans/history", headers={"Authorization": f"Bearer {token}"})
    assert history.status_code == 200
    saved = history.json()
    assert len(saved) == 1
    assert saved[0]["modality"] == "xray"
    assert saved[0]["predicted_class"] == resp.json()["predicted_class"]


def test_predict_anonymous_does_not_save_history(client, xray_sample_path, signed_up_user):
    token, _, _ = signed_up_user
    with open(xray_sample_path, "rb") as f:
        resp = client.post("/predict/xray", files={"file": ("sample.png", f, "image/png")})
    assert resp.status_code == 200

    history = client.get("/scans/history", headers={"Authorization": f"Bearer {token}"})
    assert history.json() == []


def test_predict_unverified_user_gets_403_and_nothing_saved(client, xray_sample_path, signed_up_user):
    """A signed-in user who hasn't clicked the verification link must be
    blocked from predictions (auth_routes.require_verified_if_logged_in),
    and the blocked request must not leave a scan in their history."""
    token, _, _ = signed_up_user
    with open(xray_sample_path, "rb") as f:
        resp = client.post(
            "/predict/xray",
            files={"file": ("sample.png", f, "image/png")},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 403
    assert "verify your email" in resp.json()["detail"].lower()

    history = client.get("/scans/history", headers={"Authorization": f"Bearer {token}"})
    assert history.status_code == 200
    assert history.json() == []


def test_chat_unverified_user_gets_403(client, signed_up_user):
    token, _, _ = signed_up_user
    resp = client.post(
        "/chat",
        json={"message": "What does the heatmap show?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


def test_predict_wrong_modality_rejected_by_gate(client, xray_sample_path):
    """A real X-ray sent to the CT route should be caught by the modality
    gate (backend/main.py's _gate_predict) before it ever reaches the CT
    classifier."""
    with open(xray_sample_path, "rb") as f:
        resp = client.post("/predict/ct", files={"file": ("sample.png", f, "image/png")})
    assert resp.status_code == 422
    assert "chest CT scan" in resp.json()["detail"] or "X-ray" in resp.json()["detail"]


def test_predict_rejects_blank_image(client):
    blank = Image.new("L", (256, 256), color=128)
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post("/predict/xray", files={"file": ("blank.png", buf, "image/png")})
    assert resp.status_code == 400
    assert "blank" in resp.json()["detail"].lower()


def test_predict_rejects_colorful_non_scan_image(client):
    """A colorful illustration (e.g. a "COVID-19" graphic) isn't a real
    X-ray/CT -- those are grayscale at the source -- and should be caught
    by the color-content check in _read_and_validate_image before it ever
    reaches the sanity gate or classifier."""
    colorful = Image.new("RGB", (256, 256), color=(30, 30, 30))
    for x in range(256):
        for y in range(256):
            if (x - 128) ** 2 + (y - 128) ** 2 < 100 ** 2:
                colorful.putpixel((x, y), (200, 40, 40))
    buf = io.BytesIO()
    colorful.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post("/predict/xray", files={"file": ("covid_graphic.png", buf, "image/png")})
    assert resp.status_code == 400
    assert "color" in resp.json()["detail"].lower()


def test_predict_rejects_non_image_file(client):
    resp = client.post(
        "/predict/xray",
        files={"file": ("not_an_image.txt", io.BytesIO(b"this is not an image"), "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_rejects_empty_file(client):
    resp = client.post("/predict/xray", files={"file": ("empty.png", io.BytesIO(b""), "image/png")})
    assert resp.status_code == 400


def test_predict_modality_gate_uncertain_guardrail(client, xray_sample_path, monkeypatch):
    """When the modality gate's top-class confidence is below the threshold,
    it must return HTTP 422 with 'Modality uncertain'."""
    import main

    def mock_low_conf_gate(input_tensor):
        return "xray", 0.75, {"ct": 0.25, "xray": 0.75}, {"ct": -0.5, "xray": 0.5}

    monkeypatch.setattr(main, "_gate_predict", mock_low_conf_gate)

    with open(xray_sample_path, "rb") as f:
        resp = client.post("/predict/xray", files={"file": ("sample.png", f, "image/png")})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "Modality uncertain" in detail
    assert "chest X-ray" in detail


def test_lung_crop_not_called_when_modality_gate_rejects(client, xray_sample_path, monkeypatch):
    """Verifies that the U-Net lung segmentation crop (compute_lung_crop) is
    NOT invoked when the modality gate rejects an upload (wrong modality or uncertain)."""
    import main
    import seg_utils
    crop_invoked = False

    def mock_compute_lung_crop(image, seg_bundle):
        nonlocal crop_invoked
        crop_invoked = True
        return seg_utils.compute_lung_crop(image, seg_bundle)

    monkeypatch.setattr(main, "compute_lung_crop", mock_compute_lung_crop)

    # 1. Test wrong-modality rejection (modality gate detects CT on /predict/xray)
    monkeypatch.setattr(
        main,
        "_gate_predict",
        lambda tensor: ("ct", 0.98, {"ct": 0.98, "xray": 0.02}, {"ct": 4.0, "xray": -4.0}),
    )
    with open(xray_sample_path, "rb") as f:
        resp = client.post("/predict/xray", files={"file": ("sample.png", f, "image/png")})
    assert resp.status_code == 422
    assert not crop_invoked, "compute_lung_crop was unexpectedly called on wrong modality rejection"

    # 2. Test uncertain-confidence rejection (< 0.90)
    crop_invoked = False
    monkeypatch.setattr(
        main,
        "_gate_predict",
        lambda tensor: ("xray", 0.70, {"ct": 0.30, "xray": 0.70}, {"ct": -0.8, "xray": 0.8}),
    )
    with open(xray_sample_path, "rb") as f:
        resp = client.post("/predict/xray", files={"file": ("sample.png", f, "image/png")})
    assert resp.status_code == 422
    assert not crop_invoked, "compute_lung_crop was unexpectedly called on uncertain modality rejection"
