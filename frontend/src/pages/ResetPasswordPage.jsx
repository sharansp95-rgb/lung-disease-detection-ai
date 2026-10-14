import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useAuth } from "../AuthContext";
import { ApiError } from "../api";
import AuthLayout from "../components/AuthLayout";
import Spinner from "../components/Spinner";

export default function ResetPasswordPage() {
  const { token } = useParams();
  const { resetPassword } = useAuth();
  const navigate = useNavigate();
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Passwords don't match.");
      return;
    }

    setLoading(true);
    try {
      await resetPassword(token, password);
      setDone(true);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  if (done) {
    return (
      <AuthLayout>
          <h1 className="auth-title">Password updated</h1>
          <p className="auth-subtitle">You can now log in with your new password.</p>
          <button className="btn btn-primary btn-block" onClick={() => navigate("/login")}>
            Go to log in
          </button>
        </AuthLayout>
  );
  }

  return (
    <AuthLayout>
        <h1 className="auth-title">Set a new password</h1>
        <form onSubmit={handleSubmit} className="auth-form">
          <label>
            New password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              autoFocus
            />
            <span className="input-hint">At least 8 characters</span>
          </label>
          <label>
            Confirm new password
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              minLength={8}
            />
          </label>
          {error && <div className="auth-error">{error}</div>}
          <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
            {loading ? <><Spinner /> Updating</> : "Update password"}
          </button>
        </form>
        <p className="auth-footer">
          <Link to="/login">Back to log in</Link>
        </p>
      </AuthLayout>
  );
}
