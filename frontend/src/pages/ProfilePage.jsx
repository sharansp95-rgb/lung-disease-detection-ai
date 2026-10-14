import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";
import AppNavbar from "../components/AppNavbar";
import HistoryGrid from "../components/HistoryGrid";
import { getScanHistory, ApiError } from "../api";

export default function ProfilePage() {
  const { user, token, logout } = useAuth();
  const navigate = useNavigate();

  const [scans, setScans] = useState(null);
  const [scansError, setScansError] = useState(null);

  useEffect(() => {
    getScanHistory(token)
      .then(setScans)
      .catch((e) => setScansError(e instanceof ApiError ? e.message : "Couldn't load your history."));
  }, [token]);

  function handleLogout() {
    // Navigate off this guarded route first. Clearing the session while
    // still mounted under RequireAuth races its own redirect-to-/login
    // effect against this call, and that effect can win -- so defer this
    // navigate to run after it, guaranteeing we land on the landing page.
    logout();
    setTimeout(() => navigate("/", { replace: true }), 0);
  }

  if (!user) return null;
  const initial = (user.display_name || user.email)[0].toUpperCase();
  const counts = scans
    ? { total: scans.length, xray: scans.filter((s) => s.modality === "xray").length, ct: scans.filter((s) => s.modality === "ct").length }
    : null;

  return (
    <div className="app">
      <AppNavbar />

      <main className="page">
        <header className="page-header">
          <h1 className="page-title">Profile</h1>
        </header>

        <section className="profile-account-card">
          <div className="profile-avatar profile-avatar-lg" aria-hidden="true">{initial}</div>
          <div className="profile-account-info">
            <div className="profile-account-name">{user.display_name || "No name set"}</div>
            <div className="profile-account-email">{user.email}</div>
            {counts && (
              <div className="profile-account-stats">
                {counts.total} {counts.total === 1 ? "scan" : "scans"} analysed: {counts.xray} X-ray, {counts.ct} CT
              </div>
            )}
          </div>
          <button className="btn btn-ghost" onClick={handleLogout}>Log out</button>
        </section>

        <section className="profile-section">
          <div className="profile-section-head">
            <h2 className="profile-section-title">Recent scans</h2>
            {scans && scans.length > 6 && <Link to="/history" className="text-link">See all {scans.length}</Link>}
          </div>
          <HistoryGrid scans={scans} error={scansError} limit={6} />
        </section>
      </main>
    </div>
  );
}
