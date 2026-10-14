import uuid


def _unique_email():
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def test_signup_then_me(client):
    email = _unique_email()
    resp = client.post("/auth/signup", json={"email": email, "password": "correct-horse-battery-staple"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user"]["email"] == email
    assert body["user"]["email_verified"] is False
    token = body["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_signup_duplicate_email_rejected(client):
    email = _unique_email()
    first = client.post("/auth/signup", json={"email": email, "password": "correct-horse-battery-staple"})
    assert first.status_code == 200

    dupe = client.post("/auth/signup", json={"email": email, "password": "another-password-here"})
    assert dupe.status_code == 409


def test_signup_rejects_short_password(client):
    resp = client.post("/auth/signup", json={"email": _unique_email(), "password": "short"})
    assert resp.status_code == 422


def test_signup_rejects_malformed_email(client):
    resp = client.post("/auth/signup", json={"email": "not-an-email", "password": "correct-horse-battery-staple"})
    assert resp.status_code == 422


def test_login_success(client, signed_up_user):
    _, email, password = signed_up_user
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == email


def test_login_wrong_password_rejected(client, signed_up_user):
    _, email, _ = signed_up_user
    resp = client.post("/auth/login", json={"email": email, "password": "totally-wrong-password"})
    assert resp.status_code == 401


def test_login_unknown_email_rejected(client):
    resp = client.post("/auth/login", json={"email": _unique_email(), "password": "whatever-password"})
    assert resp.status_code == 401


def test_me_without_token_rejected(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_me_with_garbage_token_rejected(client):
    resp = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_change_password_success_and_relogin(client, signed_up_user):
    token, email, password = signed_up_user
    new_password = "a-brand-new-password-123"

    resp = client.post(
        "/auth/change-password",
        json={"current_password": password, "new_password": new_password},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text

    old_login = client.post("/auth/login", json={"email": email, "password": password})
    assert old_login.status_code == 401

    new_login = client.post("/auth/login", json={"email": email, "password": new_password})
    assert new_login.status_code == 200


def test_change_password_wrong_current_rejected(client, signed_up_user):
    token, _, _ = signed_up_user
    resp = client.post(
        "/auth/change-password",
        json={"current_password": "not-the-real-password", "new_password": "a-brand-new-password-123"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_change_password_requires_auth(client):
    resp = client.post(
        "/auth/change-password",
        json={"current_password": "x", "new_password": "a-brand-new-password-123"},
    )
    assert resp.status_code == 401


def test_scan_history_empty_for_new_user(client, signed_up_user):
    token, _, _ = signed_up_user
    resp = client.get("/scans/history", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == []
