import { useAuth } from "../AuthContext";
import AuthLayout from "./AuthLayout";

// Shown in place of the app for a logged-in user whose email isn't verified
// yet. No app route (upload, chat, history, profile) is reachable until they
// click the link -- see App.jsx. Updates automatically once verified, via
// the same cross-tab broadcast SignupPage's old "Check your email" screen
// used (see EmailVerifiedPage.jsx / AuthContext.jsx).
export default function VerifyEmailGate() {
  const { user, logout } = useAuth();

  return (
    <AuthLayout>
        <h1 className="auth-title">Verify your email</h1>
        <p className="auth-subtitle">
          We sent a verification link to <strong>{user?.email}</strong>. Click it to
          unlock the app.
        </p>
        <button className="btn btn-secondary btn-block" onClick={logout}>
          Log out
        </button>
      </AuthLayout>
  );
}
