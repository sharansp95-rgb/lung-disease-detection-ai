# Auth setup: Google OAuth & Gmail SMTP

Both are optional at runtime -- the backend logs a clear warning and falls
back gracefully (console-printed links for email, a 501 for Google login)
if these aren't configured. This doc is what to do to make them real.

## Google OAuth ("Continue with Google")

**Client ID is already set** in `backend/.env` (`GOOGLE_CLIENT_ID`). Still
needed: **`GOOGLE_CLIENT_SECRET`**.

Steps in [Google Cloud Console](https://console.cloud.google.com/):

1. Open the project that owns this client ID (APIs & Services → Credentials).
2. Find the OAuth 2.0 Client ID configured in your local `backend/.env`
   and click it (or create a new "Web application" OAuth client if needed).
3. Under **Authorized redirect URIs**, add exactly:
   ```
   http://127.0.0.1:8000/auth/google/callback
   ```
   (This is `BACKEND_URL` + `/auth/google/callback` from `backend/.env` --
   if you change `BACKEND_URL`, update this too. For the public ngrok setup
   also add `https://<domain>/api/auth/google/callback`; see
   `docs/DEPLOYMENT_PUBLIC.md`.)
4. Copy the **Client secret** shown on that page.
5. Add it to `backend/.env`:
   ```
   GOOGLE_CLIENT_SECRET=<paste here>
   ```
6. Restart the backend. Startup log should show
   `Google OAuth is configured (client ID + secret present).`

Once set, `GET /auth/google/login` redirects to Google's consent screen;
after consent, Google redirects back to the callback above, which creates
or logs in the user and redirects the browser to
`{FRONTEND_URL}/oauth-callback?token=<jwt>`.

## Gmail SMTP (verification + password-reset emails)

1. The Gmail account needs 2-Step Verification turned on
   (myaccount.google.com/security).
2. Create an **App Password**: myaccount.google.com/apppasswords → select
   "Mail" (or "Other") → generate. Copy the 16-character password.
3. Add both to `backend/.env`:
   ```
   GMAIL_ADDRESS=your.address@gmail.com
   GMAIL_APP_PASSWORD=<16-char app password, no spaces>
   ```
4. Restart the backend. Startup log should show
   `Gmail SMTP is configured (...) -- verification/reset emails will send for real.`

Until this is set, verification and password-reset links are printed to
the backend console/log instead of emailed -- fully functional for testing,
just not delivered to a real inbox.
