import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { EMAIL_VERIFIED_BROADCAST_KEY } from "../AuthContext";
import AuthLayout from "../components/AuthLayout";

// Landing target for the backend's /auth/verify/{token} redirect -- this is
// the tab the EMAIL LINK opens (usually a new tab), not the original tab
// that was showing SignupPage's "Check your email" screen. On mount, it
// broadcasts a localStorage `storage` event so that original tab (and any
// other open tab) can react live without polling -- see AuthContext.jsx.
export default function EmailVerifiedPage() {
  const navigate = useNavigate();

  useEffect(() => {
    // The value doesn't matter -- writing at all is what fires `storage` in
    // other same-origin tabs. This tab won't see its own event.
    localStorage.setItem(EMAIL_VERIFIED_BROADCAST_KEY, String(Date.now()));
  }, []);

  return (
    <AuthLayout>
        <div className="verified-card">
        <div className="verified-check" aria-hidden="true">
          <svg viewBox="0 0 52 52">
            <circle className="verified-check-circle" cx="26" cy="26" r="24" fill="none" />
            <path className="verified-check-mark" fill="none" d="M14 27l8 8 16-16" />
          </svg>
        </div>
        <h1 className="auth-title">Email verified</h1>
        <p className="auth-subtitle">Your account is confirmed. You're all set.</p>
        <button className="btn btn-primary btn-block" onClick={() => navigate("/")}>
          Continue to the app
        </button>
      </div>
    </AuthLayout>
  );
}
