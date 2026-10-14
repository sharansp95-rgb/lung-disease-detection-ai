import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ResultsDisplay from "../components/ResultsDisplay";
import ChatWidget from "../components/ChatWidget";
import AppNavbar from "../components/AppNavbar";

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, previewUrl } = location.state || {};

  useEffect(() => {
    if (!result) navigate("/", { replace: true });
  }, [result, navigate]);

  if (!result) return null;

  return (
    <div className="app">
      <AppNavbar />

      <header className="page-header page-header-row">
        <h1 className="page-title">Result</h1>
        <button
          className="btn btn-ghost"
          onClick={() => navigate("/", { state: { modality: result.modality } })}
        >
          Analyse another scan
        </button>
      </header>

      <main className="app-main">
        <ResultsDisplay result={result} rawPreviewUrl={previewUrl} />
      </main>

      <ChatWidget
        predictionContext={{
          modality: result.modality,
          predicted_class: result.predicted_class,
          confidence: result.confidence,
        }}
      />
    </div>
  );
}
