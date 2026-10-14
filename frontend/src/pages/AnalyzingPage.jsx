import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { predict, ApiError } from "../api";
import { useAuth } from "../AuthContext";
import AppNavbar from "../components/AppNavbar";

const STEPS = [
  "Checking the scan type",
  "Running the classifier",
  "Drawing the heatmap",
  "Writing the explanation",
];

// Real progress isn't exposed by the backend (one request/response), so
// steps 1-3 advance on a timed simulation. The last step never resolves on
// its own -- it stays active until the actual API call finishes, so we
// never claim "done" before the real result exists.
const STEP_ADVANCE_DELAYS_MS = [900, 1900, 3000];

export default function AnalyzingPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { token } = useAuth();
  const { file, modality, previewUrl } = location.state || {};

  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!file) navigate("/", { replace: true, state: { modality } });
  }, [file, navigate, modality]);

  useEffect(() => {
    if (!file) return;

    setError(null);
    setActiveStep(0);

    const timers = STEP_ADVANCE_DELAYS_MS.map((delay, i) =>
      setTimeout(() => setActiveStep(i + 1), delay)
    );

    // AbortController, not just a `cancelled` flag: in dev, StrictMode
    // mounts this effect, cleans it up, then mounts it again -- a plain
    // flag only suppresses the first invocation's navigate/setError, the
    // real POST it already sent still completes and (for a logged-in user)
    // still saves a scan-history row, so every scan showed up twice.
    // Aborting the in-flight request on cleanup stops that first request
    // from ever reaching the backend.
    const controller = new AbortController();
    predict(modality, file, token, controller.signal)
      .then((result) => {
        navigate("/results", { replace: true, state: { result, previewUrl } });
      })
      .catch((e) => {
        if (e.name === "AbortError") return;
        timers.forEach(clearTimeout);
        setError(e instanceof ApiError ? e.message : "Something went wrong during the analysis. Try again.");
      });

    return () => {
      controller.abort();
      timers.forEach(clearTimeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, modality, token, navigate, previewUrl, attempt]);

  if (!file) return null;

  return (
    <div className="app">
      <AppNavbar />

      <main className="analyzing-page">
        <div className={`analyzing-frame ${error ? "is-error" : ""}`}>
          <span className="upload-corner upload-corner-tl" />
          <span className="upload-corner upload-corner-tr" />
          <span className="upload-corner upload-corner-bl" />
          <span className="upload-corner upload-corner-br" />
          {!error && <div className="analyzing-scanline" />}
          <img src={previewUrl} alt="Uploaded scan" className="analyzing-image" />
        </div>

        <div className="analyzing-side" aria-live="polite">
          {error ? (
            <div className="analyzing-error">
              <h1 className="analyzing-title">This image wasn't analysed</h1>
              <p>{error}</p>
              <div className="analyzing-error-actions">
                <button className="btn btn-primary" onClick={() => setAttempt((a) => a + 1)}>
                  Try again
                </button>
                <button className="btn btn-ghost" onClick={() => navigate("/", { state: { modality } })}>
                  Choose another image
                </button>
              </div>
            </div>
          ) : (
            <>
              <h1 className="analyzing-title">Analysing your scan</h1>
              <ol className="analyzing-steps">
                {STEPS.map((label, i) => {
                  const state = i < activeStep ? "done" : i === activeStep ? "active" : "pending";
                  return (
                    <li className={`analyzing-step ${state}`} key={label}>
                      <span className="analyzing-step-icon">
                        {state === "done" && (
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                            <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        )}
                      </span>
                      <span>{label}</span>
                    </li>
                  );
                })}
              </ol>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
