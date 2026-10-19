import { useEffect, useRef, useState } from "react";

// Same figures as the "Internal vs External Accuracy" slide in the final review.
const GROUPS = [
  {
    name: "Chest X-ray",
    internal: { value: 96.5, note: "Validation images" },
    external: { value: 93.9, note: "New images (31 of 33)" },
  },
  {
    name: "Lung CT",
    internal: { value: 99.0, note: "Validation images" },
    external: { value: 91.9, note: "New images" },
  },
];

const SERIES = [
  { key: "internal", label: "Internal (validation)", color: "var(--chart-internal)" },
  { key: "external", label: "External data", color: "var(--chart-external)" },
];

const TICKS = [0, 25, 50, 75, 100];

export default function AccuracyChart() {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  // Bars grow once, when the chart scrolls into view.
  useEffect(() => {
    const el = ref.current;
    if (!el || !("IntersectionObserver" in window)) {
      setVisible(true);
      return undefined;
    }
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          io.disconnect();
        }
      },
      { threshold: 0.35 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <>
      <div ref={ref} className={`accuracy ${visible ? "is-visible" : ""}`}>
        <figure className="acc-chart">
          <figcaption className="acc-chart-title">Detection accuracy (%)</figcaption>
          <div className="acc-plot" aria-hidden="true">
            <div className="acc-axis">
              {TICKS.map((t) => (
                <span key={t} style={{ bottom: `${t}%` }}>{t}</span>
              ))}
            </div>
            <div className="acc-area">
              {TICKS.slice(1).map((t) => (
                <span key={t} className="acc-grid" style={{ bottom: `${t}%` }} />
              ))}
              {GROUPS.map((g, gi) => (
                <div className="acc-group" key={g.name}>
                  {SERIES.map((s, si) => (
                    <span
                      key={s.key}
                      className="acc-bar"
                      tabIndex={0}
                      style={{ "--v": `${g[s.key].value}%`, "--c": s.color, "--d": `${(gi * 2 + si) * 0.12}s` }}
                    >
                      <span className="acc-bar-value">{g[s.key].value.toFixed(1)}</span>
                      <span className="acc-tip">
                        {g.name}, {s.label.toLowerCase()}: {g[s.key].value.toFixed(1)}% ({g[s.key].note.toLowerCase()})
                      </span>
                    </span>
                  ))}
                </div>
              ))}
            </div>
          </div>
          <div className="acc-labels" aria-hidden="true">
            <span />
            {GROUPS.map((g) => <span key={g.name}>{g.name}</span>)}
          </div>
          <ul className="acc-legend">
            {SERIES.map((s) => (
              <li key={s.key}><i style={{ "--c": s.color }} />{s.label}</li>
            ))}
          </ul>
          <div className="sr-only">
          <table className="acc-table">
            <caption>Detection accuracy, internal vs external</caption>
            <thead><tr><th>Scan</th><th>Internal (validation)</th><th>External data</th></tr></thead>
            <tbody>
              {GROUPS.map((g) => (
                <tr key={g.name}><td>{g.name}</td><td>{g.internal.value}%</td><td>{g.external.value}%</td></tr>
              ))}
            </tbody>
          </table>
          </div>
        </figure>

        <dl className="acc-stats">
          {GROUPS.flatMap((g) => [
            <div className="acc-stat" key={`${g.name}-i`}>
              <dt>{g.name} internal</dt>
              <dd>
                <span className="acc-stat-value">{g.internal.value.toFixed(1)}%</span>
                <span className="acc-stat-note">{g.internal.note}</span>
              </dd>
            </div>,
            <div className={`acc-stat ${g.external.value < 80 ? "acc-stat-low" : ""}`} key={`${g.name}-e`}>
              <dt>{g.name} external</dt>
              <dd>
                <span className="acc-stat-value">{g.external.value.toFixed(1)}%</span>
                <span className="acc-stat-note">{g.external.note}</span>
              </dd>
            </div>,
          ])}
        </dl>
      </div>
      <p className="acc-takeaway">
        <strong>Takeaway:</strong> X-ray stays accurate on new data. CT drops sharply because it
        learned cues from a single dataset.
      </p>
    </>
  );
}
