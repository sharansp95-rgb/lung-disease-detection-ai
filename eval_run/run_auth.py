"""Section 1F: full auth + scan-history flow against the QA instance (port 8001,
email disabled so verification/reset links go to the server console, isolated DB)."""
import json, pathlib, re, time, sys
import requests

B = "http://127.0.0.1:8001"
ROOT = pathlib.Path(r"d:/7th sem/sdp")
LOG = ROOT / "eval_run" / "backend2_startup.log"
out = {}
email = f"qa.tester.{int(time.time())}@example.test"
PW1, PW2, PW3 = "QaPassw0rd!1", "QaPassw0rd!2", "QaPassw0rd!3"


def step(name, r):
    try:
        b = r.json()
    except Exception:
        b = {"raw": r.text[:400]}
    out[name] = {"status": r.status_code, "body": b}
    print(f"== {name}: {r.status_code} {json.dumps(b)[:220]}", flush=True)
    return b


def find_link(kind, since):
    txt = LOG.read_text(encoding="utf-8", errors="replace")[since:]
    pat = r"http://[^\s]*/auth/verify/[A-Za-z0-9_\-]+" if kind == "verify" else r"http://[^\s]*/reset-password/[A-Za-z0-9_\-]+"
    m = re.findall(pat, txt)
    return m[-1] if m else None


mark = len(LOG.read_text(encoding="utf-8", errors="replace"))

# 1. signup
b = step("signup", requests.post(f"{B}/auth/signup", json={"email": email, "password": PW1, "name": "QA Tester"}))
token = b.get("access_token")
out["signup_issues_token_pre_verification"] = bool(token)
out["signup_user_email_verified_flag"] = b.get("user", {}).get("email_verified")

# 2. token usable before verification?
step("me_pre_verification", requests.get(f"{B}/auth/me", headers={"Authorization": f"Bearer {token}"}))
step("predict_pre_verification_allowed", requests.post(
    f"{B}/predict/ct", headers={"Authorization": f"Bearer {token}"},
    files={"file": ("c.png", (ROOT / "data/ct_scan/test/Normal").glob("*").__next__().read_bytes(), "image/png")}))

# 3. verification link from console
time.sleep(1)
vlink = find_link("verify", mark)
out["verification_link_found"] = bool(vlink)
if vlink:
    step("verify_click", requests.get(vlink, allow_redirects=False))
    step("verify_click_again", requests.get(vlink, allow_redirects=False))
step("me_post_verification", requests.get(f"{B}/auth/me", headers={"Authorization": f"Bearer {token}"}))

# 4. login
b = step("login_original_pw", requests.post(f"{B}/auth/login", json={"email": email, "password": PW1}))
token = b.get("access_token", token)

# 5. change-password
step("change_pw_wrong_current", requests.post(f"{B}/auth/change-password",
     json={"current_password": "wrongwrong1", "new_password": PW2},
     headers={"Authorization": f"Bearer {token}"}))
step("change_pw_ok", requests.post(f"{B}/auth/change-password",
     json={"current_password": PW1, "new_password": PW2},
     headers={"Authorization": f"Bearer {token}"}))
step("login_old_pw_after_change", requests.post(f"{B}/auth/login", json={"email": email, "password": PW1}))
b = step("login_new_pw_after_change", requests.post(f"{B}/auth/login", json={"email": email, "password": PW2}))
token = b.get("access_token", token)
step("old_session_token_still_valid_after_pw_change",
     requests.get(f"{B}/auth/me", headers={"Authorization": f"Bearer {token}"}))

# 6. forgot / reset
mark2 = len(LOG.read_text(encoding="utf-8", errors="replace"))
step("forgot_password", requests.post(f"{B}/auth/forgot-password", json={"email": email}))
step("forgot_password_unknown_email", requests.post(f"{B}/auth/forgot-password", json={"email": "nobody@example.test"}))
time.sleep(1)
rlink = find_link("reset", mark2)
out["reset_link_found"] = bool(rlink)
if rlink:
    rtok = rlink.rsplit("/", 1)[-1]
    step("reset_reuse_current_pw_rejected", requests.post(f"{B}/auth/reset-password/{rtok}", json={"new_password": PW2}))
    step("reset_ok", requests.post(f"{B}/auth/reset-password/{rtok}", json={"new_password": PW3}))
    step("reset_token_reuse", requests.post(f"{B}/auth/reset-password/{rtok}", json={"new_password": "AnotherPw12!"}))
step("login_pw2_after_reset", requests.post(f"{B}/auth/login", json={"email": email, "password": PW2}))
b = step("login_pw3_after_reset", requests.post(f"{B}/auth/login", json={"email": email, "password": PW3}))
token = b.get("access_token", token)

# 7. authenticated prediction -> history exact match
ct_file = sorted((ROOT / "data/ct_scan/test/Malignant").glob("*"))[0]
r = requests.post(f"{B}/predict/ct", headers={"Authorization": f"Bearer {token}"},
                  files={"file": (ct_file.name, ct_file.read_bytes(), "image/png")})
pred = r.json()
out["auth_prediction"] = {"status": r.status_code, "predicted_class": pred.get("predicted_class"),
                          "confidence": pred.get("confidence"), "probs": pred.get("class_probabilities"),
                          "heatmap_len": len(pred.get("heatmap_png_base64") or ""),
                          "report_len": len(pred.get("report") or ""), "report_source": pred.get("report_source")}
h = requests.get(f"{B}/scans/history", headers={"Authorization": f"Bearer {token}"})
hist = h.json()
out["history_status"] = h.status_code
out["history_count"] = len(hist) if isinstance(hist, list) else None
if isinstance(hist, list) and hist:
    item = hist[0]
    out["history_item_keys"] = sorted(item.keys())
    cmp = {
        "predicted_class": (item.get("predicted_class") == pred.get("predicted_class")),
        "confidence_exact": (item.get("confidence") == pred.get("confidence")),
        "confidence_values": [item.get("confidence"), pred.get("confidence")],
        "probs_exact": (item.get("class_probabilities") == pred.get("class_probabilities")),
        "probs_values": [item.get("class_probabilities"), pred.get("class_probabilities")],
        "report_exact": (item.get("report") == pred.get("report")),
        "heatmap_exact": (item.get("heatmap_png_base64") == pred.get("heatmap_png_base64")),
        "heatmap_present_in_history": bool(item.get("heatmap_png_base64")),
        "report_source_exact": (item.get("report_source") == pred.get("report_source")),
        "modality": item.get("modality"),
    }
    out["history_vs_response"] = cmp
    print("history compare:", json.dumps({k: v for k, v in cmp.items() if not k.endswith("values")}, indent=1), flush=True)

# 8. anonymous prediction persists nothing
before = len(requests.get(f"{B}/scans/history", headers={"Authorization": f"Bearer {token}"}).json())
requests.post(f"{B}/predict/ct", files={"file": (ct_file.name, ct_file.read_bytes(), "image/png")})
after_r = requests.get(f"{B}/scans/history", headers={"Authorization": f"Bearer {token}"})
out["anon_prediction_history_delta"] = {"before": before, "after": len(after_r.json())}

# 9. history requires auth
step("history_no_token", requests.get(f"{B}/scans/history"))
step("history_bad_token", requests.get(f"{B}/scans/history", headers={"Authorization": "Bearer garbage"}))

# 10. google oauth
r = requests.get(f"{B}/auth/google/login", allow_redirects=False)
out["google_login"] = {"status": r.status_code, "location_host": (r.headers.get("location") or "")[:80]}
print("google login:", out["google_login"], flush=True)

(ROOT / "eval_run" / "results_auth.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print("SAVED", flush=True)
