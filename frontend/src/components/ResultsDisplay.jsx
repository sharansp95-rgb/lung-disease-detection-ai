import { useState } from "react";
import useCountUp from "../hooks/useCountUp";

const CLASS_COLORS = {
  "COVID-19": "var(--cls-covid)",
  Pneumonia: "var(--cls-pneumonia)",
  Tuberculosis: "var(--cls-tb)",
  Malignant: "var(--cls-malignant)",
  Benign: "var(--cls-benign)",
  Normal: "var(--cls-normal)",
};

const MODALITY_NAMES = { xray: "chest X-ray", ct: "chest CT slice" };

function ProbabilityRow({ name, prob, index, isTop }) {
  const shown = useCountUp(prob * 100, 700 + index * 120);
  return (
    <div className={`prob-row ${isTop ? "is-top" : ""}`}>
      <span className="prob-class">{name}</span>
      <div className="prob-bar-track">
        <div
          className="prob-bar-fill"
          style={{ width: `${shown}%`, background: CLASS_COLORS[name] || "var(--text)" }}
        />
      </div>
      <span className="prob-value">{shown.toFixed(1)}%</span>
    </div>
  );
}

export default function ResultsDisplay({ result, rawPreviewUrl }) {
  const [view, setView] = useState("heatmap");
  const confidence = useCountUp(result.confidence * 100, 900);

  const sortedClasses = Object.entries(result.class_probabilities).sort((a, b) => b[1] - a[1]);
  const hasWarnings = result.warnings && result.warnings.length > 0;
  const color = CLASS_COLORS[result.predicted_class] || "var(--text)";

  return (
    <div className="results">
      <section className="result-summary" aria-labelledby="result-class">
        <p className="result-kicker">
          Model prediction for this {MODALITY_NAMES[result.modality] || "scan"}
        </p>
        <h2 id="result-class" className="result-class" style={{ "--class-color": color }}>
          <span className="result-class-dot" aria-hidden="true" />
          {result.predicted_class}
        </h2>
        <p className="result-confidence">
          <span className="result-confidence-number">{confidence.toFixed(1)}%</span> model confidence
        </p>

        {hasWarnings && (
          <div className="warning-banner" role="status">
            {result.warnings.map((w, i) => (
              <p key={i}>{w}</p>
            ))}
          </div>
        )}

        <div className="prob-breakdown">
          <h3 className="section-label">Probability of each class</h3>
          {sortedClasses.map(([className, prob], i) => (
            <ProbabilityRow key={className} name={className} prob={prob} index={i} isTop={i === 0} />
          ))}
        </div>
      </section>

      <section className="result-panel result-panel-image" aria-label="Heatmap">
        <div className="result-panel-head">
          <h3 className="section-label">Where the model looked</h3>
          <div className="image-toggle" role="tablist" aria-label="Image view">
            <button role="tab" aria-selected={view === "heatmap"} className={view === "heatmap" ? "active" : ""} onClick={() => setView("heatmap")}>
              Heatmap
            </button>
            <button role="tab" aria-selected={view === "original"} className={view === "original" ? "active" : ""} onClick={() => setView("original")}>
              Original
            </button>
          </div>
        </div>
        <div className="results-image-stack">
          <img src={rawPreviewUrl} alt="Original upload" className={`results-image ${view === "original" ? "is-visible" : ""}`} />
          <img src={result.heatmap_png_base64} alt="Grad-CAM heatmap over the scan" className={`results-image ${view === "heatmap" ? "is-visible" : ""}`} />
        </div>
        <div className="heatmap-legend" aria-hidden={view !== "heatmap"}>
          <span>Less influence</span>
          <span className="heatmap-legend-bar" />
          <span>More influence</span>
        </div>
        <p className="result-note">
          The heatmap shows the regions that most influenced the model's output. It is not a
          marking of disease.
        </p>
      </section>

      <section className="result-panel result-panel-report" aria-label="Explanation">
        <h3 className="section-label">Explanation</h3>
        <p className="report-text">{result.report}</p>
        <p className="report-source">
          Written by {result.report_source === "groq" ? "a language model (Groq)" : "the built-in template"}
        </p>
        <div className="disclaimer-box">
          This is an automated model output, not a medical diagnosis. Always consult a qualified
          radiologist or physician.
        </div>
      </section>
    </div>
  );
}
