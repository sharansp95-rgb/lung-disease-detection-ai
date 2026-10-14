import { Link } from "react-router-dom";
import ThemeToggle from "../components/ThemeToggle";
import BrandMark from "../components/BrandMark";
import GuardTrace from "../components/GuardTrace";
import AccuracyChart from "../components/AccuracyChart";

const STEPS = [
  {
    title: "Checked before anything else",
    desc: "Blank files, colour photos, documents, screenshots and the wrong scan type are refused with a reason you can read.",
  },
  {
    title: "Cropped to the lungs",
    desc: "Chest X-rays are cropped to the lung fields by a segmentation model, so the classifier sees what it was trained on.",
  },
  {
    title: "Classified",
    desc: "A DenseNet-121 model gives the probability of every class, not just the top one.",
  },
  {
    title: "Explained",
    desc: "A heatmap shows where the model looked, and a short report explains the result in plain words.",
  },
];

export default function LandingPage() {
  return (
    <div className="landing">
      <header className="landing-nav">
        <BrandMark />
        <div className="landing-nav-actions">
          <ThemeToggle />
          <Link to="/login" className="btn btn-ghost">Log in</Link>
          <Link to="/signup" className="btn btn-primary">Create account</Link>
        </div>
      </header>

      <section className="landing-hero">
        <div className="landing-hero-content">
          <h1 className="landing-hero-title">
            A second reader for chest X-rays and CT slices.
          </h1>
          <p className="landing-hero-subtitle">
            Upload a scan to get a model prediction, a heatmap of where the model looked and a
            plain-language explanation. If the image isn't something it was trained on, it tells
            you so instead of guessing.
          </p>
          <div className="landing-hero-actions">
            <Link to="/signup" className="btn btn-primary btn-lg">Create a free account</Link>
            <Link to="/login" className="btn btn-ghost btn-lg">I already have one</Link>
          </div>
        </div>
        <GuardTrace />
      </section>

      <section className="landing-section" aria-labelledby="how-title">
        <h2 id="how-title" className="landing-section-title">What happens to an upload</h2>
        <ol className="landing-steps">
          {STEPS.map((s, i) => (
            <li className="landing-step" key={s.title}>
              <span className="landing-step-index" aria-hidden="true">{i + 1}</span>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="landing-section" aria-labelledby="accuracy-title">
        <h2 id="accuracy-title" className="landing-section-title">Internal vs external accuracy</h2>
        <AccuracyChart />
      </section>

      <footer className="landing-footer">
        <p>
          This is a research prototype for decision support, not a diagnostic service. Always consult a
          qualified radiologist or physician.
        </p>
      </footer>
    </div>
  );
}
