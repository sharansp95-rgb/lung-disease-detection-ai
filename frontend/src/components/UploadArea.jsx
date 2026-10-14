import { useRef, useState } from "react";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // mirrors the backend's limit, for fast client-side feedback only

export default function UploadArea({ file, previewUrl, onFileSelected, onError, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function validateAndSet(selected) {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      onError("That file doesn't look like an image. Please choose an image file.");
      return;
    }
    if (selected.size > MAX_UPLOAD_BYTES) {
      onError(
        `File is too large (${(selected.size / (1024 * 1024)).toFixed(1)}MB). Maximum allowed is 10MB.`
      );
      return;
    }
    onFileSelected(selected);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const dropped = e.dataTransfer.files?.[0];
    validateAndSet(dropped);
  }

  return (
    <div
      className={`upload-area ${dragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Add a scan image"
    >
      <span className="upload-corner upload-corner-tl" />
      <span className="upload-corner upload-corner-tr" />
      <span className="upload-corner upload-corner-bl" />
      <span className="upload-corner upload-corner-br" />
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        disabled={disabled}
        onChange={(e) => validateAndSet(e.target.files?.[0])}
      />
      {previewUrl ? (
        <img src={previewUrl} alt="Selected scan" className="upload-preview" key={previewUrl} />
      ) : (
        <div className="upload-placeholder">
          <svg className="upload-icon" width="34" height="34" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 15V4m0 0L7.5 8.5M12 4l4.5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          <p className="upload-title">{dragging ? "Drop to add this image" : "Drop an image here, or choose a file"}</p>
          <p className="upload-hint">One scan per analysis</p>
        </div>
      )}
      {file && <p className="upload-filename">{file.name} — click to replace</p>}
    </div>
  );
}
