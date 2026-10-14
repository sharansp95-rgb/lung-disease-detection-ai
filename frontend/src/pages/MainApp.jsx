import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ModalitySelector from "../components/ModalitySelector";
import UploadArea from "../components/UploadArea";
import ChatWidget from "../components/ChatWidget";
import AppNavbar from "../components/AppNavbar";

export default function MainApp({ backendStatus }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [modality, setModality] = useState(location.state?.modality || "xray");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [error, setError] = useState(null);

  function handleModalityChange(m) {
    setModality(m);
    setFile(null);
    setPreviewUrl(null);
    setError(null);
  }

  function handleFileSelected(selected) {
    setError(null);
    setFile(selected);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(selected);
    });
  }

  function handleUploadError(message) {
    setError(message);
  }

  function handleSubmit() {
    if (!file) return;
    navigate("/analyzing", { state: { file, modality, previewUrl } });
  }

  return (
    <div className="app">
      <AppNavbar />

      <header className="page-header">
        <h1 className="page-title">Analyse a scan</h1>
        <p className="page-subtitle">
          Choose the scan type, then add one image. PNG or JPEG, up to 10 MB.
        </p>
      </header>

      {backendStatus.checked && !backendStatus.reachable && (
        <div className="backend-warning">
          Can't reach the analysis server. Start the backend, then reload this page.
        </div>
      )}

      <main className="app-main">
        <ModalitySelector
          modality={modality}
          onChange={handleModalityChange}
          disabled={false}
        />

        <section className="upload-panel">
          <UploadArea
            file={file}
            previewUrl={previewUrl}
            onFileSelected={handleFileSelected}
            onError={handleUploadError}
            disabled={false}
          />

          <button
            className="submit-btn"
            onClick={handleSubmit}
            disabled={!file}
          >
            {modality === "ct" ? "Analyse CT slice" : "Analyse chest X-ray"}
          </button>

          {error && <div className="error-banner" role="alert">{error}</div>}
        </section>
      </main>

      <ChatWidget predictionContext={null} />
    </div>
  );
}
