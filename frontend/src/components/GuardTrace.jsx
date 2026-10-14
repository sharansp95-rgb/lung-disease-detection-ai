// The landing page's one animated moment: two uploads travel down the same
// four checks. The photo is refused at the second check; the X-ray passes
// all four and comes out as a prediction. This is the project's core idea
// ("refuse rather than guess"), shown instead of described.
const GATES = [
  "Readable, greyscale image",
  "Looks like a medical scan",
  "Matches the chosen scan type",
  "Close to the training data",
];

export default function GuardTrace() {
  return (
    <figure className="guard-trace" aria-label="Every upload passes four checks before it is classified. A photo is refused at the second check; a chest X-ray passes all four and receives a prediction.">
      <div className="guard-lanes" aria-hidden="true">
        <div className="guard-lane-label guard-lane-label-a">Phone photo</div>
        <div className="guard-lane-label guard-lane-label-b">Chest X-ray</div>
        <span className="guard-token guard-token-a" />
        <span className="guard-token guard-token-b" />
        <ol className="guard-gates">
          {GATES.map((g, i) => (
            <li key={g} className={`guard-gate guard-gate-${i + 1}`}>
              <span className="guard-gate-light" />
              {g}
            </li>
          ))}
        </ol>
        <div className="guard-outcomes">
          <div className="guard-outcome guard-outcome-refused">
            Refused: this doesn't look like a chest X-ray
          </div>
          <div className="guard-outcome guard-outcome-result">
            <span>Prediction</span>
            <strong>Pneumonia</strong>
            <span className="guard-outcome-bar"><i /></span>
          </div>
        </div>
      </div>
      <figcaption className="guard-caption">Four checks run before any prediction.</figcaption>
    </figure>
  );
}
