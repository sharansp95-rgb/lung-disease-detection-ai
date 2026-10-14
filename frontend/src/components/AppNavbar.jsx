import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";
import ThemeToggle from "./ThemeToggle";
import BrandMark from "./BrandMark";

export default function AppNavbar() {
  const { user } = useAuth();
  const navigate = useNavigate();

  if (!user) return null;

  const initial = (user.display_name || user.email)[0].toUpperCase();

  return (
    <header className="app-navbar">
      <BrandMark />
      <nav className="app-navbar-links" aria-label="Main">
        <NavLink to="/" end className="nav-link">Analyse</NavLink>
        <NavLink to="/history" className="nav-link">History</NavLink>
      </nav>
      <div className="app-navbar-actions">
        <ThemeToggle />
        <button
          className="profile-avatar"
          onClick={() => navigate("/profile")}
          aria-label="Profile"
          title="Profile"
        >
          {initial}
        </button>
      </div>
    </header>
  );
}
