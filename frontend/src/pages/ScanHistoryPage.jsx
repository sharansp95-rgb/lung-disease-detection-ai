import { useEffect, useState } from "react";
import { useAuth } from "../AuthContext";
import { getScanHistory, ApiError } from "../api";
import AppNavbar from "../components/AppNavbar";
import HistoryGrid from "../components/HistoryGrid";

export default function ScanHistoryPage() {
  const { token } = useAuth();
  const [scans, setScans] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getScanHistory(token)
      .then(setScans)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Couldn't load your history. Reload the page to try again."));
  }, [token]);

  return (
    <div className="app">
      <AppNavbar />
      <main className="page">
        <header className="page-header">
          <h1 className="page-title">History</h1>
          <p className="page-subtitle">
            Every scan you've analysed, with the heatmap and explanation exactly as you saw them.
          </p>
        </header>
        <HistoryGrid scans={scans} error={error} />
      </main>
    </div>
  );
}
