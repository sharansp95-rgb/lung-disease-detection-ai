import BrandMark from "./BrandMark";
import ThemeToggle from "./ThemeToggle";

// Shared frame for every sign-in style page: a quiet statement panel on the
// left (hidden on phones) and the form on the right.
export default function AuthLayout({ children, aside = true }) {
  return (
    <div className="auth-page">
      {aside && (
        <aside className="auth-aside">
          <BrandMark />
          <div className="auth-aside-body">
            <p className="auth-aside-quote">
              Every upload is checked before it is classified. Photos, documents and the wrong
              scan type are refused, not guessed at.
            </p>
            <ul className="auth-aside-list">
              <li>Grad-CAM heatmap with every prediction</li>
              <li>Plain-language explanation</li>
              <li>Your scan history, saved to your account</li>
            </ul>
          </div>
          <p className="auth-aside-foot">Research prototype for decision support. Not a diagnostic service.</p>
        </aside>
      )}
      <main className="auth-main">
        <div className="auth-main-top">
          {!aside && <BrandMark />}
          <ThemeToggle />
        </div>
        <div className="auth-card">{children}</div>
      </main>
    </div>
  );
}
