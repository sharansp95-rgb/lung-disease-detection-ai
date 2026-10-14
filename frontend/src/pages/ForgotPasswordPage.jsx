import { useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../AuthContext";
import { ApiError } from "../api";
import AuthLayout from "../components/AuthLayout";
import Spinner from "../components/Spinner";

export default function ForgotPasswordPage() {
  const { forgotPassword } = useAuth();
  const [email, setEmail] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await forgotPassword(email);
      setSent(true);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout>
        <h1 className="auth-title">Reset your password</h1>

        {sent ? (
          <p className="auth-subtitle">
            If an account exists for <strong>{email}</strong>, a reset link has been sent.
          </p>
        ) : (
          <>
            <p className="auth-subtitle">Enter your email and we'll send you a reset link.</p>
            <form onSubmit={handleSubmit} className="auth-form">
              <label>
                Email
                <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
              </label>
              {error && <div className="auth-error">{error}</div>}
              <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
                {loading ? <><Spinner /> Sending</> : "Send reset link"}
              </button>
            </form>
          </>
        )}

        <p className="auth-footer">
          <Link to="/login">Back to log in</Link>
        </p>
      </AuthLayout>
  );
}
