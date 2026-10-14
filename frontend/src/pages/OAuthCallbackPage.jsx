import { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../AuthContext";
import AuthLayout from "../components/AuthLayout";
import Spinner from "../components/Spinner";

export default function OAuthCallbackPage() {
  const [searchParams] = useSearchParams();
  const { loginWithToken } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const token = searchParams.get("token");
    if (token) {
      loginWithToken(token);
      navigate("/");
    } else {
      navigate("/login?error=google_no_token");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <AuthLayout>
        <p className="auth-subtitle"><Spinner /> Signing you in</p>
      </AuthLayout>
  );
}
