import { Link } from "react-router-dom";

// Viewfinder brackets around a pair of lungs: the same bracket motif is used
// on the upload area and the analysing screen, so the brand and the product
// share one visual idea.
export function BrandGlyph({ size = 28 }) {
  return (
    <svg className="brand-glyph" width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <path d="M3 10V5a2 2 0 0 1 2-2h5M22 3h5a2 2 0 0 1 2 2v5M29 22v5a2 2 0 0 1-2 2h-5M10 29H5a2 2 0 0 1-2-2v-5"
        stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M16 8v7" stroke="var(--signal)" strokeWidth="2" strokeLinecap="round" />
      <path d="M13.5 12.5c-3.2 0-5 2.6-5 6.6 0 3 1.2 4.9 3.1 4.9 1.7 0 2.4-1.2 2.3-3.6-.1-2.1-.4-4-.4-7.9ZM18.5 12.5c3.2 0 5 2.6 5 6.6 0 3-1.2 4.9-3.1 4.9-1.7 0-2.4-1.2-2.3-3.6.1-2.1.4-4 .4-7.9Z"
        stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
  );
}

export default function BrandMark({ to = "/" }) {
  return (
    <Link to={to} className="brand" aria-label="Lung Disease Detection, home">
      <BrandGlyph />
      <span className="brand-name">Lung Disease Detection</span>
    </Link>
  );
}
