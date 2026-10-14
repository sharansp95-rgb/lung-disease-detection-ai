import { createContext, useContext, useEffect, useState } from "react";
import { API_BASE_URL, ApiError } from "./api";

const AuthContext = createContext(null);
const TOKEN_KEY = "auth_token";
// Cross-tab signal only -- the value itself carries no meaning, writing it
// is what fires the browser's built-in `storage` event in OTHER same-origin
// tabs (never the tab that wrote it), which is how the tab that's still on
// the "Check your email" screen finds out verification happened in the tab
// the email link opened, with no polling and no new dependency.
export const EMAIL_VERIFIED_BROADCAST_KEY = "email_verified_broadcast";

async function apiPost(path, body, token) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(`Cannot reach the backend server at ${API_BASE_URL}. Is it running?`);
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(`Backend returned an unreadable response (HTTP ${response.status}).`, response.status);
  }
  if (!response.ok) {
    const detail = data?.detail || `Request failed (HTTP ${response.status}).`;
    throw new ApiError(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail, response.status);
  }
  return data;
}

async function apiGet(path, token) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { headers });
  } catch {
    throw new ApiError(`Cannot reach the backend server at ${API_BASE_URL}. Is it running?`);
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(`Backend returned an unreadable response (HTTP ${response.status}).`, response.status);
  }
  if (!response.ok) {
    throw new ApiError(data?.detail || `Request failed (HTTP ${response.status}).`, response.status);
  }
  return data;
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) {
      setUser(null);
      setLoading(false);
      return;
    }
    apiGet("/auth/me", token)
      .then(setUser)
      .catch(() => {
        // token invalid/expired -- clear it
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, [token]);

  // When another tab verifies the user's email (SignupPage's "Check your
  // email" screen sits in this tab; the email link opens verification in a
  // NEW tab), that tab writes EMAIL_VERIFIED_BROADCAST_KEY to localStorage.
  // This tab never sees its own writes via `storage` -- only other tabs do
  // -- so re-fetching /auth/me here picks up the fresh email_verified flag
  // without this tab needing to poll or reload.
  useEffect(() => {
    function handleStorage(event) {
      if (event.key !== EMAIL_VERIFIED_BROADCAST_KEY || !event.newValue) return;
      const currentToken = localStorage.getItem(TOKEN_KEY);
      if (!currentToken) return;
      apiGet("/auth/me", currentToken).then(setUser).catch(() => {});
    }
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  function setSession(accessToken, userObj) {
    localStorage.setItem(TOKEN_KEY, accessToken);
    setToken(accessToken);
    setUser(userObj);
  }

  async function signup(email, password, name) {
    const data = await apiPost("/auth/signup", { email, password, name: name || null });
    setSession(data.access_token, data.user);
    return data;
  }

  async function login(email, password) {
    const data = await apiPost("/auth/login", { email, password });
    setSession(data.access_token, data.user);
    return data;
  }

  function loginWithToken(accessToken) {
    localStorage.setItem(TOKEN_KEY, accessToken);
    setToken(accessToken);
  }

  function logout() {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
  }

  async function forgotPassword(email) {
    return apiPost("/auth/forgot-password", { email });
  }

  async function resetPassword(resetToken, newPassword) {
    return apiPost(`/auth/reset-password/${resetToken}`, { new_password: newPassword });
  }

  return (
    <AuthContext.Provider
      value={{
        token,
        user,
        loading,
        signup,
        login,
        loginWithToken,
        logout,
        forgotPassword,
        resetPassword,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
