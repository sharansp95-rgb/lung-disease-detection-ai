// Default talks to the backend directly (plain local dev). Set
// VITE_API_BASE_URL=/api to go through the Vite proxy instead -- that's the
// public ngrok setup, see docs/DEPLOYMENT_PUBLIC.md. Every backend call
// (including AuthContext.jsx and the Google login link) uses this base.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

export const MODALITY_CLASSES = {
  xray: ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"],
  ct: ["Benign", "Malignant", "Normal"],
};

export const MODALITY_LABELS = {
  xray: "Chest X-ray",
  ct: "Chest CT Scan",
};

/**
 * Calls POST /predict/{modality} with the given file.
 * Throws an ApiError with a clean, backend-provided message on any failure
 * (network error, non-2xx response, or unparseable response) -- never lets
 * a raw fetch/JS error reach the UI unformatted.
 */
export class ApiError extends Error {
  constructor(message, status = null) {
    super(message);
    this.status = status;
  }
}

export async function predict(modality, file, token, signal) {
  const formData = new FormData();
  formData.append("file", file);

  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/predict/${modality}`, {
      method: "POST",
      headers,
      body: formData,
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      `Cannot reach the backend server at ${API_BASE_URL}. Is it running?`
    );
  }

  let body;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(
      `Backend returned an unreadable response (HTTP ${response.status}).`,
      response.status
    );
  }

  if (!response.ok) {
    // FastAPI's HTTPException responses put the real message in `detail` --
    // use it as-is, never a generic invented message.
    const detail = body?.detail || `Request failed (HTTP ${response.status}).`;
    throw new ApiError(detail, response.status);
  }

  return body;
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) return { reachable: false };
    const body = await response.json();
    return { reachable: true, ...body };
  } catch {
    return { reachable: false };
  }
}

/**
 * Calls POST /chat. `history` is a plain array of {role, content} (no UI
 * metadata). `predictionContext` is {modality, predicted_class, confidence}
 * or null if no result is currently loaded. Same ApiError discipline as
 * predict() -- clean backend message on failure, never a raw JS error.
 */
export async function sendChatMessage(message, history, predictionContext, token) {
  let response;
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        message,
        history,
        prediction_context: predictionContext || null,
      }),
    });
  } catch {
    throw new ApiError(
      `Cannot reach the backend server at ${API_BASE_URL}. Is it running?`
    );
  }

  let body;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(
      `Backend returned an unreadable response (HTTP ${response.status}).`,
      response.status
    );
  }

  if (!response.ok) {
    const detail = body?.detail || `Request failed (HTTP ${response.status}).`;
    throw new ApiError(detail, response.status);
  }

  return body;
}

export async function getScanHistory(token) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/scans/history`, {
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    throw new ApiError(`Cannot reach the backend server at ${API_BASE_URL}. Is it running?`);
  }
  let body;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(`Backend returned an unreadable response (HTTP ${response.status}).`, response.status);
  }
  if (!response.ok) {
    throw new ApiError(body?.detail || `Request failed (HTTP ${response.status}).`, response.status);
  }
  return body;
}

export function googleLoginUrl() {
  return `${API_BASE_URL}/auth/google/login`;
}

export { API_BASE_URL };
