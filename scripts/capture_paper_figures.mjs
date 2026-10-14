// Captures the six real-app screenshots (F1-F6) used in the IEEE paper, in
// LIGHT MODE, from the actually running app (backend on :8000, frontend Vite
// dev server on :5173 -- both must already be started before running this).
//
// Usage (from repo root):
//   node scripts/capture_paper_figures.mjs <output_dir>
//
// Forces light mode by setting localStorage BEFORE navigation (an init
// script), rather than clicking the in-app theme toggle, so there is no
// dark-mode flash and no race with ThemeContext's own effect.
//
// Uses a fresh, clearly-fake demo account (no real personal email) created
// via the real /auth/signup endpoint -- not a mock -- and does not touch or
// clear the existing backend/app.db.

import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// @playwright/test lives under frontend/node_modules, not repo-root
// node_modules (there is none) -- resolve it explicitly by absolute path so
// this script can be invoked from anywhere with a plain system `node`.
const playwrightEntry = pathToFileURL(
  path.join(ROOT, "frontend", "node_modules", "@playwright", "test", "index.mjs")
).href;
const { chromium } = await import(playwrightEntry);
const OUT_DIR = process.argv[2] || path.join(ROOT, "paper_figures");
mkdirSync(OUT_DIR, { recursive: true });

const FRONTEND_URL = "http://localhost:5173";
const BACKEND_URL = "http://127.0.0.1:8000";
const DEMO_EMAIL = `demo.reviewer.${Date.now()}@example.com`;
const DEMO_PASSWORD = "paper-demo-account-1234";

// Tuberculosis-1.png is excluded: it is a colorized variant that the
// backend's own color-content guard correctly rejects (verified separately);
// -10/-11 are genuine greyscale scans that pass validation.
const XRAY_TB_SAMPLE = path.join(ROOT, "sample images", "tb", "Tuberculosis-10.png");
const XRAY_TB_SAMPLE_2 = path.join(ROOT, "sample images", "tb", "Tuberculosis-11.png");
const CT_MALIGNANT_SAMPLE = path.join(ROOT, "sample images", "cancer", "malignant cases", "Malignant case (1).jpg");
const CT_MALIGNANT_SAMPLE_2 = path.join(ROOT, "sample images", "cancer", "malignant cases", "Malignant case (2).jpg");

async function signupDemoAccount() {
  const resp = await fetch(`${BACKEND_URL}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: DEMO_EMAIL, password: DEMO_PASSWORD, name: "Demo Reviewer" }),
  });
  if (!resp.ok) throw new Error(`signup failed: ${resp.status} ${await resp.text()}`);
  const body = await resp.json();
  console.log(`[signup] created ${DEMO_EMAIL}, user id ${body.user.id}`);
  return body.access_token;
}

async function forceLightMode(context) {
  await context.addInitScript(() => {
    try {
      window.localStorage.setItem("theme", "light");
    } catch (e) {}
    // hide scrollbars on every document this context loads, applied at
    // DOMContentLoaded since <head> may not exist yet when this runs
    document.addEventListener("DOMContentLoaded", () => {
      const style = document.createElement("style");
      style.textContent = "::-webkit-scrollbar { display: none; } html { scrollbar-width: none; }";
      document.head.appendChild(style);
    });
  });
}

async function setAuthToken(page, token) {
  await page.addInitScript((t) => {
    try {
      window.localStorage.setItem("auth_token", t);
    } catch (e) {}
  }, token);
}

async function uploadAndAnalyze(page, modality, filePath) {
  await page.goto(`${FRONTEND_URL}/`, { waitUntil: "networkidle" });
  if (modality === "ct") {
    await page.getByRole("button", { name: /CT/i }).click();
  }
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(filePath);
  await page.getByRole("button", { name: /^Analyze/ }).click();
  await page.waitForURL(/\/results$/, { timeout: 45000 });
  await page.waitForSelector(".report-text", { state: "visible", timeout: 45000 });
  // report text is filled once the Groq/template call resolves; wait until non-empty
  await page.waitForFunction(() => {
    const el = document.querySelector(".report-text");
    return el && el.textContent && el.textContent.trim().length > 10;
  }, { timeout: 45000 });
  await page.waitForTimeout(400);
}

async function main() {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  await forceLightMode(context);
  const page = await context.newPage();

  // ---- F1: landing page (viewport tall enough that the disclaimer at the
  // bottom of the hero/how-it-works flow is visible without scrolling) ----
  await page.goto(`${FRONTEND_URL}/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(300);
  const disclaimerBox = await page.locator(".landing-disclaimer").boundingBox();
  if (disclaimerBox && disclaimerBox.y + disclaimerBox.height > 880) {
    const neededHeight = Math.min(Math.ceil(disclaimerBox.y + disclaimerBox.height + 40), 2200);
    await page.setViewportSize({ width: 1440, height: neededHeight });
    await page.waitForTimeout(200);
  }
  await page.screenshot({ path: path.join(OUT_DIR, "F1_landing_page.png") });
  console.log("[F1] captured landing page");
  await page.setViewportSize({ width: 1440, height: 900 });

  // ---- demo account + auth ----
  const token = await signupDemoAccount();
  await setAuthToken(page, token);

  // ---- F2: X-ray result (Tuberculosis, confident + localized) ----
  await uploadAndAnalyze(page, "xray", XRAY_TB_SAMPLE);
  await page.screenshot({ path: path.join(OUT_DIR, "F2_xray_result.png"), fullPage: true });
  console.log("[F2] captured X-ray result");

  // second X-ray prediction, purely to populate scan history for F4
  await uploadAndAnalyze(page, "xray", XRAY_TB_SAMPLE_2);

  // ---- F3: CT result (Malignant) ----
  await uploadAndAnalyze(page, "ct", CT_MALIGNANT_SAMPLE);
  await page.screenshot({ path: path.join(OUT_DIR, "F3_ct_result.png"), fullPage: true });
  console.log("[F3] captured CT result");

  // second CT prediction, to populate scan history for F4
  await uploadAndAnalyze(page, "ct", CT_MALIGNANT_SAMPLE_2);

  // ---- F4: profile page with saved scan history ----
  await page.goto(`${FRONTEND_URL}/profile`, { waitUntil: "networkidle" });
  await page.waitForSelector(".history-grid", { state: "visible", timeout: 15000 });
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT_DIR, "F4_profile_history.png"), fullPage: true });
  console.log("[F4] captured profile/history page");

  // ---- F5: chat widget with FAQ exchange ----
  await page.goto(`${FRONTEND_URL}/`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /open chat/i }).click();
  await page.waitForSelector(".chat-panel", { state: "visible" });
  await page.getByRole("button", { name: "What does the heatmap show?" }).click();
  await page.waitForFunction(() => {
    const bubbles = document.querySelectorAll(".chat-message.assistant .chat-bubble");
    return bubbles.length > 0 && bubbles[bubbles.length - 1].textContent.trim().length > 10;
  }, { timeout: 30000 });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT_DIR, "F5_chat_widget.png") });
  console.log("[F5] captured chat widget");

  // ---- F6: rejection state (wrong-modality: CT image submitted on the X-ray route) ----
  await page.goto(`${FRONTEND_URL}/`, { waitUntil: "networkidle" });
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(CT_MALIGNANT_SAMPLE);
  await page.getByRole("button", { name: /^Analyze/ }).click();
  await page.waitForURL(/\/analyzing$/, { timeout: 15000 });
  await page.waitForSelector(".analyzing-error", { state: "visible", timeout: 20000 });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT_DIR, "F6_rejection_state.png") });
  console.log("[F6] captured rejection state");

  await browser.close();
  console.log("done.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
