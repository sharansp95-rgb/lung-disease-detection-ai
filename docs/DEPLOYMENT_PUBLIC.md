# Public access via ngrok

This guide makes the app reachable from any device (a phone on mobile data, a tablet, another computer) at a fixed public URL like `https://<name>.ngrok-free.app`. Everything still runs on this Windows laptop. There is no cloud hosting, no Docker and no LAN exposure.

Plain local development (`uvicorn main:app --reload` plus `npm run dev`) is unchanged. Every setting below is opt-in, and each one has a localhost default.

---

## How it works

```
Phone / any browser
   │  https://<domain>
   ▼
ngrok cloud  ──tunnel──►  ngrok agent (this laptop)
                              │  http://127.0.0.1:4173
                              ▼
                     vite preview (built frontend)
                       ├─ /            → static app (dist/)
                       └─ /api/*       → strips /api → http://127.0.0.1:8000
                                              ▼
                                   FastAPI backend (127.0.0.1 only)
```

- **One port is exposed:** 4173, the frontend. The backend listens only on `127.0.0.1:8000`, and the outside world reaches it only through the `/api` proxy.
- **The browser sees one origin**, so the public app never makes cross-origin requests.
- **Forwarding headers are kept:**
  - The proxy leaves the original `Host` header and ngrok's `X-Forwarded-Proto` unchanged.
  - It appends its own hop to `X-Forwarded-For`.
  - The backend's rate limiter (`backend/rate_limit.py`) reads the visitor's real IP from `X-Forwarded-For`, but **only** when the direct peer is `127.0.0.1`. Without that, every visitor would share one rate-limit bucket.

---

## 1. One-time setup

### 1.1 ngrok account and agent

1. **Sign up** (free) at <https://dashboard.ngrok.com/signup>.
2. **Install the agent on Windows.** Use one of the options on ngrok's Windows download page (<https://ngrok.com/downloads/windows>):
   - Microsoft Store: search for "ngrok". This option updates itself, and ngrok recommends it because Windows Defender sometimes flags a downloaded `ngrok.exe` by mistake.
   - WinGet: `winget install ngrok -s msstore`
   - Scoop: `scoop install ngrok`
   - The ZIP download. If you use this, put `ngrok.exe` in a folder on your `PATH`.
3. **Check the install.** Open a new PowerShell window and run `ngrok version`.
4. **Add your authtoken.**
   1. Copy it from <https://dashboard.ngrok.com/get-started/your-authtoken>.
   2. Run:
      ```powershell
      ngrok config add-authtoken <YOUR_AUTHTOKEN>
      ```
   3. This writes the token to ngrok's own config file, `%LocalAppData%\ngrok\ngrok.yml`, which is **outside this repo**. Never put the authtoken in the repo, in `.env` files or in scripts.
   4. To check the config, run `ngrok config check`.
5. **Find your dev domain.** Every free account gets one automatically assigned dev domain. It can't be chosen or changed, and it stays the same.
   - It is listed in the dashboard under **Domains** (<https://dashboard.ngrok.com/domains>).
   - It ends in `.ngrok-free.app` or `.ngrok-free.dev`, depending on when your account was created. Use exactly what the dashboard shows.
   - The rest of this guide calls it `<domain>`, for example `your-assigned-name.ngrok-free.app`.

### 1.2 Backend settings: `backend/.env`

Add or replace these lines. Keep your existing secrets as they are.

```dotenv
# Default public URLs, used when the request's origin isn't in
# ALLOWED_PUBLIC_ORIGINS.
FRONTEND_URL=https://<domain>
BACKEND_URL=https://<domain>/api

# Origins the app can be used from at the same time. Email links and Google
# sign-in follow the origin the user is on if it's listed here.
ALLOWED_PUBLIC_ORIGINS=https://<domain>,http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173

# Browser origins allowed by CORS (comma-separated). Behind the /api proxy,
# requests are same-origin, but listing the public origin is harmless. Keep
# the localhost entries so plain local dev keeps working.
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://<domain>
```

| Variable | Default (plain local dev) | Public value |
|---|---|---|
| `FRONTEND_URL` | `http://localhost:5173` | `https://<domain>` |
| `BACKEND_URL` | `http://127.0.0.1:8000` | `https://<domain>/api` |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | add `https://<domain>` |

| `ALLOWED_PUBLIC_ORIGINS` | `FRONTEND_URL` + localhost `:5173`/`:4173` | same, or list explicitly |

> **Local and public at the same time:** no need to switch `.env`. Each allowed origin needs its own Google redirect URI: `https://<domain>/api/auth/google/callback`, `http://localhost:4173/api/auth/google/callback` (and `127.0.0.1:4173`), and `http://127.0.0.1:8000/auth/google/callback` for `npm run dev`, which calls the backend directly.

The backend prints the values it loaded at startup. It never prints secrets. Look for these lines:

```
FRONTEND_URL = https://<domain>
BACKEND_URL = https://<domain>/api
Google OAuth redirect_uri = https://<domain>/api/auth/google/callback
CORS origins = ...
```

### 1.3 Frontend settings: `frontend/.env.public.local`

Copy `frontend/.env.public.example` to `frontend/.env.public.local`. The `.local` file is gitignored. Then set:

```dotenv
VITE_API_BASE_URL=/api
FRONTEND_ALLOWED_HOSTS=<domain>
```

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Base URL for **every** backend call: `api.js`, `AuthContext.jsx` and the "Continue with Google" link. `/api` sends them through the proxy. |
| `FRONTEND_ALLOWED_HOSTS` | *(empty: Vite accepts only localhost and IP addresses)* | Extra `Host` headers Vite accepts (comma-separated hostnames). |
| `BACKEND_PROXY_TARGET` | `http://127.0.0.1:8000` | Optional. Where `/api/*` is forwarded. |

This file is loaded only in `public` mode (`vite build --mode public` and `vite preview --mode public`), which `start_public.ps1` uses. `npm run dev` ignores it.

> Set these values in the file, not by exporting them from Git Bash. Git Bash rewrites values that look like paths, such as `/api`.

### 1.4 Google Cloud Console (only if you use "Continue with Google")

1. Go to <https://console.cloud.google.com/> → **APIs & Services → Credentials** and open the OAuth 2.0 Client ID that the app uses (the one in `GOOGLE_CLIENT_ID`).
2. Under **Authorized JavaScript origins**, add:
   ```
   https://<domain>
   ```
3. Under **Authorized redirect URIs**, add exactly:
   ```
   https://<domain>/api/auth/google/callback
   ```
   This must match `BACKEND_URL` + `/auth/google/callback` character for character: `https`, no trailing slash, and `/api` included.
4. Keep the existing `http://127.0.0.1:8000/auth/google/callback` entry so plain local dev still works.
5. Click **Save**. Google says changes can take a few minutes, sometimes longer, to take effect.
6. If the OAuth consent screen is still in **Testing** status, only the Google accounts listed as test users can sign in. Add the accounts you'll demo with.

---

## 2. Start and stop

From the repo root, in PowerShell:

```powershell
.\start_public.ps1
```

If PowerShell refuses to run scripts, use this for the current window only: `Set-ExecutionPolicy -Scope Process Bypass`.

### What the script does

1. **Checks** that `python`, `node` and `ngrok` are installed, that the frontend env file is set up, and that ports 8000 and 4173 are free. It warns if `FRONTEND_URL` or `BACKEND_URL` in `backend/.env` don't match the domain.
2. **Builds** the frontend in public mode.
3. **Starts the backend** on `127.0.0.1:8000`. Logs go to `logs/public_backend.*.log`.
4. **Waits** until `GET http://127.0.0.1:8000/health` returns 200. This **takes more than a minute** because of the model reproduction checks, and dots are printed while it waits.
5. **Starts `vite preview`** on `127.0.0.1:4173`. Logs go to `logs/public_preview.*.log`.
6. **Runs** `ngrok http 127.0.0.1:4173 --url https://<domain>` in the same window. The ngrok status screen shows the forwarding URL and live requests.

**To stop:** press **Ctrl+C** in that window. This stops ngrok, vite preview and the backend, and the public URL goes offline right away.

If something was left running (for example, the window was force-closed), run:

```powershell
.\start_public.ps1 -Stop
```

This stops any running `ngrok` process and whatever is listening on ports 4173 and 8000.

Optional parameters: `-BackendPort 8000`, `-FrontendPort 4173`, `-HealthTimeoutSeconds 600`. You can also set `$env:NGROK_DOMAIN` to override the domain that is normally read from `FRONTEND_ALLOWED_HOSTS`.

---

## 3. Test from a phone on mobile data

Testing on the laptop itself, or on the same Wi-Fi network, doesn't prove the public path works. To test properly:

1. **Turn Wi-Fi off** on the phone so it uses mobile data only.
2. **Open** `https://<domain>` in the phone's browser.
3. **Get past the ngrok warning page** by tapping **Visit Site** (see §4).
4. **Test an upload:** upload a chest X-ray and check that the prediction, heatmap and report appear.
5. **Test the chat:** send a chat message.
6. **Test sign-up and email:**
   1. Sign up.
   2. Open the verification email **on the phone**. Its link should start with `https://<domain>/api/auth/verify/`.
   3. Check that the link lands on `https://<domain>/email-verified`.
7. **Test password reset:** use "Forgot password". The reset link should be `https://<domain>/reset-password/...`.
8. **Test Google sign-in:** tap "Continue with Google". After consent, you should land back on `https://<domain>/oauth-callback` and be logged in.

---

## 4. The ngrok browser warning page

On the free plan, ngrok shows a warning page ("You are about to visit...") the first time a browser opens the URL. It is an anti-phishing measure that ngrok applies to free endpoints. After the visitor clicks **Visit Site**, that browser doesn't see it again for about 7 days. The only ways to remove it entirely are a paid plan, or a client that sends an `ngrok-skip-browser-warning` header (normal browser navigation can't send custom headers).

**Before a demo, open the URL on every demo device and click through the warning.** Do it in the same browser app you'll use for the demo, because the click-through is remembered per browser. The page can also appear when Google or an email link sends the browser back to the site, if that browser hasn't clicked through yet.

---

## 5. During demos: power and sleep

The site is only online while the laptop is awake and connected to the internet.

- **Keep the laptop plugged in.**
- **Disable sleep while plugged in:**
  1. Open Settings → System → Power & battery → Screen, sleep & hibernate timeouts.
  2. Set "When plugged in, put my device to sleep after" to **Never**.
  3. Or, from an admin PowerShell: `powercfg /change standby-timeout-ac 0`
- **Don't close the lid** unless the lid-close action is set to "Do nothing" (Control Panel → Power Options → Choose what closing the lid does).
- **Start the script at least 3–5 minutes before the demo** so the backend's startup checks finish.

---

## 6. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| **Blank page, or "Blocked request. This host (...) is not allowed."** | `FRONTEND_ALLOWED_HOSTS` in `frontend/.env.public.local` doesn't match the domain exactly. Use the hostname only, with no `https://`. Restart the script. A blank page can also mean the build used the wrong mode: always start with `start_public.ps1`, not plain `npm run preview`. |
| **"Cannot reach the backend server at http://127.0.0.1:8000"** on a phone | The build didn't pick up `VITE_API_BASE_URL=/api`. Check `frontend/.env.public.local` and restart the script, which rebuilds the frontend. |
| **CORS errors in the browser console** | With the `/api` proxy, requests are same-origin, so a CORS error almost always means the frontend is calling the backend directly (see the previous row). If you really need direct calls, add the exact origin (scheme + host, no trailing slash) to `CORS_ORIGINS` in `backend/.env` and restart. |
| **Google: `Error 400: redirect_uri_mismatch`** | The redirect URI in Google Cloud Console must equal `BACKEND_URL` + `/auth/google/callback`, for example `https://<domain>/api/auth/google/callback`. Compare it with the `Google OAuth redirect_uri = ...` line in the backend startup log. Changes can take a few minutes to apply. |
| **Google: `access_denied` / "app has not completed verification"** | The consent screen is in Testing status and the account isn't a test user. Add the account under the OAuth consent screen's test users. |
| **Email links point to `127.0.0.1` or `localhost`** | `FRONTEND_URL` or `BACKEND_URL` in `backend/.env` wasn't updated, or the backend wasn't restarted. The script prints a warning when they don't match. |
| **HTTP 413 "File exceeds maximum allowed size of 10MB"** | The backend rejects uploads over 10 MB on purpose. The proxy and ngrok pass the body through, so compress or resize the image. |
| **ngrok error page `ERR_NGROK_8012` (502)** | The tunnel works, but nothing is answering on `127.0.0.1:4173`. vite preview isn't running: check `logs/public_preview.err.log` and restart the script. |
| **App shows "Backend returned an unreadable response (HTTP 502)"** | vite preview is up but the backend isn't. The proxy returns 502 and `logs/public_preview.out.log` shows `http proxy error`. Check `logs/public_backend.err.log`. |
| **Uploads fail right after starting** | The backend is still running its startup reproduction checks, which take more than a minute. The script doesn't start ngrok until `/health` returns 200. If you started the pieces by hand, wait for `Startup complete` in the backend log. |
| **HTTP 429 "Rate limit exceeded"** | Per-visitor limits: `/predict/*` 20/min, `/chat` 20/min, `/auth/login` 20/min, `/auth/signup` 30/hour, `/auth/forgot-password` 5/hour. Limits are counted per real client IP. Several phones on the same mobile carrier can share an IP (carrier NAT). Wait a minute, or restart the backend to clear the in-memory counters. |
| **HTTP 422 on chat** | The message is over 2,000 characters or the history has more than 50 entries. The frontend already limits both. |
| **`ERR_NGROK_4018` / authentication failed** | The authtoken isn't set up. Run `ngrok config add-authtoken <YOUR_AUTHTOKEN>` again. |
| **"Port 8000/4173 is already in use"** | A previous run is still going. Run `.\start_public.ps1 -Stop`. |
| **Script stops: `Backend exited during startup`** | Read `logs/public_backend.err.log`. A failed reproduction check stops startup on purpose. |
| **Site stops responding partway through a demo** | Check the laptop's internet connection and sleep settings, and look at the ngrok window. The free plan also has monthly limits (1 GB data transfer and 20,000 HTTP requests, per ngrok's free-plan limits page). 10 MB uploads use that data allowance quickly. |

---

## 7. Security note

While `start_public.ps1` is running, **anyone who has the URL** can use:

- the whole web app: upload scans, see predictions, and use the chat, which uses **your Groq API quota**;
- every backend API route through `https://<domain>/api/...`: prediction, `/chat`, `/health`, and all `/auth/*` routes (sign-up, login, password reset, Google sign-in, change password), plus `/scans/history` for logged-in users.

What is **not** exposed:

- the backend port itself (it is bound to `127.0.0.1`);
- the source code and `.env` files (vite preview serves only the built `dist/` folder);
- the rest of the laptop.

Keep in mind:

- **Data stays on the laptop.** Accounts and scan history (for logged-in users) are saved in `backend/app.db`. Anyone who signs up through the public URL creates a real row there.
- **Abuse protection is limited.** It consists of the per-IP rate limits, the 10 MB upload cap and the chat payload caps. It is meant for a demo, not a production service.
- **This is a research prototype, not a medical device.** Don't invite real patients to upload real medical images.
- **The URL never changes.** Anyone you shared it with can come back whenever it is online.

**Closing ngrok takes the app offline immediately.** Press Ctrl+C in the script window, or run `.\start_public.ps1 -Stop`. The ngrok dashboard (<https://dashboard.ngrok.com>) shows whether an endpoint is currently online.
