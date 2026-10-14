import { MODALITY_CLASSES, MODALITY_LABELS } from "../api";

const MODALITIES = ["xray", "ct"];

// Segmented control. The highlight is one element that slides between the
// two options (driven by --index), so switching shows what changed.
export default function ModalitySelector({ modality, onChange, disabled }) {
  const index = Math.max(0, MODALITIES.indexOf(modality));
  return (
    <div className="modality-selector" role="radiogroup" aria-label="Scan type" style={{ "--index": index }}>
      <span className="modality-indicator" aria-hidden="true" />
      {MODALITIES.map((m) => (
        <button
          key={m}
          type="button"
          role="radio"
          aria-checked={modality === m}
          className={`modality-btn ${modality === m ? "active" : ""}`}
          onClick={() => onChange(m)}
          disabled={disabled}
        >
          <span className="modality-name">{MODALITY_LABELS[m]}</span>
          <span className="modality-classes">{MODALITY_CLASSES[m].join(", ")}</span>
        </button>
      ))}
    </div>
  );
}
