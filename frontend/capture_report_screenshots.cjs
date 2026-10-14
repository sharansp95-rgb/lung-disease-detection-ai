// One-off capture script for the SDP report (Chapter 7 + Appendix screenshots).
// Drives the real running frontend (localhost:5173) against the real running
// backend (127.0.0.1:8000) end to end and saves named PNGs into
// ../report/images/ScreenShots/. Not part of the app; run with:
//   node capture_report_screenshots.cjs
const { chromium } = require("@playwright/test");
const path = require("path");

const BASE = "http://localhost:5173";
const OUT = path.resolve(__dirname, "..", "report", "images", "ScreenShots");
const EMAIL = `report_demo_${Date.now()}@example.com`;
const PASSWORD = "ReportDemo123";

function shot(page, name, opts = {}) {
  return page.screenshot({ path: path.join(OUT, name), fullPage: true, ...opts });
}

async function waitForAssistantReply(page, expectedCount) {
  await page.waitForFunction(
    (n) => document.querySelectorAll(".chat-message.assistant").length >= n,
    expectedCount,
    { timeout: 30000 }
  );
  await page.locator(".typing-indicator").waitFor({ state: "detached", timeout: 30000 }).catch(() => {});
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });

  // 01. Landing page (dark, default)
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.locator(".landing-hero-title").waitFor({ state: "visible", timeout: 10000 });
  await shot(page, "01_landing_dark.png");
  console.log("01 landing (dark) OK");

  // 02. Signup page
  await page.locator('a:has-text("Sign Up")').click();
  await page.waitForURL(/signup/);
  await page.locator(".auth-card").waitFor({ state: "visible", timeout: 10000 });
  await shot(page, "02_signup_page.png");
  console.log("02 signup page OK");

  // 03. Submit signup -> verification prompt
  await page.locator('input[type="email"]').fill(EMAIL);
  await page.locator('input[type="password"]').nth(0).fill(PASSWORD);
  await page.locator('input[type="password"]').nth(1).fill(PASSWORD);
  await page.locator('button[type="submit"]').click();
  await page.locator('h1:has-text("Check your email")').waitFor({ timeout: 15000 });
  await shot(page, "03_signup_verify_prompt.png");
  console.log("03 verify prompt OK");

  // 04. Continue to app, logged in, upload screen
  await page.locator('button:has-text("Continue to the app")').click();
  await page.waitForSelector(".profile-avatar", { timeout: 10000 });
  await shot(page, "04_main_app_upload_dark.png");
  console.log("04 main app (logged in, dark) OK");

  // 05. X-ray prediction result (Tuberculosis case) with Grad-CAM heatmap
  const xrayImg = path.resolve(__dirname, "..", "data", "xray", "test", "Tuberculosis", "tbdb_Tuberculosis-104.png");
  await page.locator('input[type="file"]').setInputFiles(xrayImg);
  await page.locator(".upload-preview").waitFor({ timeout: 5000 });
  await page.locator(".submit-btn").click();
  await page.locator(".results").waitFor({ timeout: 60000 });
  await page.waitForTimeout(400);
  await shot(page, "05_xray_result_gradcam.png");
  console.log("05 X-ray result OK");

  // 06. Open chat widget grounded in this result
  await page.locator(".chat-fab").click();
  await page.locator(".chat-panel").waitFor({ state: "visible", timeout: 5000 });
  await page.locator(".chat-input-row textarea").fill("why did it think this was Tuberculosis?");
  await page.locator(".chat-input-row button").click();
  await waitForAssistantReply(page, 1);
  await shot(page, "06_chat_widget_context.png");
  console.log("06 chat widget OK");
  await page.locator(".chat-close-btn").click().catch(() => {});

  // 07. Back to upload screen, run a CT scan prediction (Malignant case)
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector(".profile-avatar", { timeout: 10000 });
  const buttons = await page.locator(".modality-btn").all();
  await buttons[1].click(); // CT
  const ctImg = path.resolve(__dirname, "..", "data", "ct_scan", "test", "Malignant", "Malignant case (109).jpg");
  await page.locator('input[type="file"]').setInputFiles(ctImg);
  await page.locator(".upload-preview").waitFor({ timeout: 5000 });
  await page.locator(".submit-btn").click();
  await page.locator(".results").waitFor({ timeout: 60000 });
  await page.waitForTimeout(400);
  await shot(page, "07_ct_result_gradcam.png");
  console.log("07 CT result OK");

  // 08. Light theme toggle
  await page.locator(".theme-toggle").click();
  await page.waitForTimeout(300);
  await shot(page, "08_main_app_light.png");
  console.log("08 light theme OK");

  // 09. Scan History page (standalone route, light theme)
  await page.goto(BASE + "/history", { waitUntil: "networkidle" });
  await page.locator(".history-card").first().waitFor({ timeout: 10000 });
  await shot(page, "09_scan_history_light.png");
  console.log("09 scan history OK");

  // 10. History detail modal
  await page.locator(".history-card").first().click();
  await page.locator(".history-modal").waitFor();
  await shot(page, "10_scan_history_modal.png");
  console.log("10 history modal OK");
  await page.locator(".history-modal-close").click().catch(() => {});

  // 11. Profile page (account info + inline history)
  await page.goto(BASE + "/profile", { waitUntil: "networkidle" });
  await page.locator(".profile-account-card").waitFor({ timeout: 10000 });
  await shot(page, "11_profile_page.png");
  console.log("11 profile page OK");

  // 12. Logout -> landing page
  await page.locator('button:has-text("Log Out")').click();
  await page.waitForSelector(".landing-hero-title", { timeout: 10000 });
  await page.waitForTimeout(300);
  await shot(page, "12_landing_after_logout.png");
  console.log("12 landing after logout OK");

  // 13. Login page
  await page.goto(BASE + "/login", { waitUntil: "networkidle" });
  await page.locator(".auth-card").waitFor({ state: "visible", timeout: 10000 });
  await shot(page, "13_login_page.png");
  console.log("13 login page OK");

  // 14. Forgot-password page
  await page.goto(BASE + "/forgot-password", { waitUntil: "networkidle" });
  await page.locator(".auth-card").waitFor({ state: "visible", timeout: 10000 });
  await shot(page, "14_forgot_password_page.png");
  console.log("14 forgot-password page OK");

  await browser.close();
  console.log("\nAll report screenshots captured into", OUT);
})().catch((err) => {
  console.error("CAPTURE FAILED:", err);
  process.exit(1);
});
