import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { MODALITY_LABELS } from "../api";

const CLASS_COLORS = {
  "COVID-19": "var(--cls-covid)",
  Pneumonia: "var(--cls-pneumonia)",
  Tuberculosis: "var(--cls-tb)",
  Malignant: "var(--cls-malignant)",
  Benign: "var(--cls-benign)",
  Normal: "var(--cls-normal)",
};

function formatDate(createdAt) {
  // SQLite returns "YYYY-MM-DD HH:MM:SS" in UTC; make it ISO so every
  // browser (Safari included) parses it.
  const d = new Date(`${createdAt.replace(" ", "T")}Z`);
  return Number.isNaN(d.getTime()) ? createdAt : d.toLocaleString();
}

function ScanModal({ scan, onClose }) {
  const closeRef = useRef(null);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="history-modal-backdrop" onClick={onClose}>
      <div
        className="history-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="scan-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <button ref={closeRef} className="icon-btn history-modal-close" onClick={onClose} aria-label="Close">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
        <img src={scan.heatmap_png_base64} alt="Saved Grad-CAM heatmap" className="history-modal-img" />
        <div className="history-modal-body">
          <p className="history-modality">{MODALITY_LABELS[scan.modality] || scan.modality}</p>
          <h2 id="scan-modal-title" style={{ "--class-color": CLASS_COLORS[scan.predicted_class] }}>
            <span className="result-class-dot" aria-hidden="true" />
            {scan.predicted_class}
          </h2>
          <p className="history-modal-conf">{(scan.confidence * 100).toFixed(1)}% model confidence</p>
          <p className="history-modal-report">{scan.report}</p>
          <p className="history-date">{formatDate(scan.created_at)}</p>
        </div>
      </div>
    </div>
  );
}

// Shared by the History and Profile pages (they used to carry two copies).
export default function HistoryGrid({ scans, error, limit }) {
  const [selected, setSelected] = useState(null);

  if (error) return <div className="auth-error" role="alert">{error}</div>;

  if (scans === null) {
    return (
      <div className="history-grid" aria-busy="true">
        {Array.from({ length: 4 }).map((_, i) => (
          <div className="history-card history-card-skeleton" key={i} />
        ))}
      </div>
    );
  }

  if (scans.length === 0) {
    return (
      <div className="history-empty">
        <p>No scans yet. Your results appear here after each analysis.</p>
        <Link to="/" className="btn btn-primary">Analyse a scan</Link>
      </div>
    );
  }

  const shown = limit ? scans.slice(0, limit) : scans;

  return (
    <>
      <div className="history-grid">
        {shown.map((s) => (
          <button key={s.id} className="history-card" onClick={() => setSelected(s)}>
            <img src={s.heatmap_png_base64} alt="" className="history-thumb" />
            <div className="history-card-info">
              <span className="history-modality">{MODALITY_LABELS[s.modality] || s.modality}</span>
              <span className="history-class" style={{ "--class-color": CLASS_COLORS[s.predicted_class] }}>
                <span className="result-class-dot" aria-hidden="true" />
                {s.predicted_class}
              </span>
              <span className="history-meta">
                {(s.confidence * 100).toFixed(1)}% · {formatDate(s.created_at)}
              </span>
            </div>
          </button>
        ))}
      </div>
      {selected && <ScanModal scan={selected} onClose={() => setSelected(null)} />}
    </>
  );
}
