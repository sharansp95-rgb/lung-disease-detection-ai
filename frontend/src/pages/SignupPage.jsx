import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";
import { ApiError } from "../api";
import AuthLayout from "../components/AuthLayout";
import Spinner from "../components/Spinner";

export default function SignupPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Passwords don't match.");
      return;
    }

    setLoading(true);
    try {
      await signup(email, password, name.trim());
      // RootRoute (App.jsx) shows VerifyEmailGate for an unverified user and
      // switches to the real app on its own once the link is clicked -- no
      // separate "check your email" screen or waiting logic needed here.
      navigate("/");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout>
        <h1 className="auth-title">Create your account</h1>
        <p className="auth-subtitle">You need an account to analyse scans and keep your history.</p>

        <form onSubmit={handleSubmit} className="auth-form">
          <label>
            Name
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} autoFocus />
          </label>
          <label>
            Email
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
            />
            <span className="input-hint">At least 8 characters</span>
          </label>
          <label>
            Confirm password
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
            {loading ? <><Spinner /> Creating account</> : "Create account"}
          </button>
        </form>

        <p className="auth-footer">
          Already have an account? <Link to="/login">Log in</Link>
        </p>
      </AuthLayout>
  );
}
