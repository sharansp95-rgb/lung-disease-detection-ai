import { useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "./AuthContext";
import { ThemeProvider } from "./ThemeContext";
import LandingPage from "./pages/LandingPage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import ForgotPasswordPage from "./pages/ForgotPasswordPage";
import ResetPasswordPage from "./pages/ResetPasswordPage";
import OAuthCallbackPage from "./pages/OAuthCallbackPage";
import EmailVerifiedPage from "./pages/EmailVerifiedPage";
import ScanHistoryPage from "./pages/ScanHistoryPage";
import ProfilePage from "./pages/ProfilePage";
import AnalyzingPage from "./pages/AnalyzingPage";
import ResultsPage from "./pages/ResultsPage";
import MainApp from "./pages/MainApp";
import VerifyEmailGate from "./components/VerifyEmailGate";
import { checkHealth } from "./api";
import "./App.css";

function RootRoute({ backendStatus }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="app-loading" />;
  if (!user) return <LandingPage />;
  if (!user.email_verified) return <VerifyEmailGate />;
  return <MainApp backendStatus={backendStatus} />;
}

function RequireAuth({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="app-loading" />;
  if (!user) return <Navigate to="/login" replace />;
  if (!user.email_verified) return <VerifyEmailGate />;
  return children;
}


// Keyed on the path so each navigation gets a short fade-in: the only
// motion that runs on every page, and it only runs in response to a click.
function AnimatedRoutes({ backendStatus }) {
  const location = useLocation();
  return (
    <div className="route-view" key={location.pathname}>
            <Routes location={location}>
              <Route path="/" element={<RootRoute backendStatus={backendStatus} />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/signup" element={<SignupPage />} />
              <Route path="/forgot-password" element={<ForgotPasswordPage />} />
              <Route path="/reset-password/:token" element={<ResetPasswordPage />} />
              <Route path="/oauth-callback" element={<OAuthCallbackPage />} />
              <Route path="/email-verified" element={<EmailVerifiedPage />} />
              <Route
                path="/history"
                element={
                  <RequireAuth>
                    <ScanHistoryPage />
                  </RequireAuth>
                }
              />
              <Route
                path="/profile"
                element={
                  <RequireAuth>
                    <ProfilePage />
                  </RequireAuth>
                }
              />
              <Route
                path="/analyzing"
                element={
                  <RequireAuth>
                    <AnalyzingPage />
                  </RequireAuth>
                }
              />
              <Route
                path="/results"
                element={
                  <RequireAuth>
                    <ResultsPage />
                  </RequireAuth>
                }
              />
            </Routes>
    </div>
  );
}

export default function App() {
  const [backendStatus, setBackendStatus] = useState({ checked: false, reachable: false });

  useEffect(() => {
    checkHealth().then((status) => setBackendStatus({ checked: true, ...status }));
  }, []);

  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <AnimatedRoutes backendStatus={backendStatus} />
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}
