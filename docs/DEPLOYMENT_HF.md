# Deploying to a Hugging Face Space

One Docker Space (free "CPU basic") serves the whole app from one origin:
the React build at `/` and the FastAPI API under `/api` (`backend/serve.py`).
User accounts and scan history live in a free hosted Postgres (Neon), because
the Space's disk is wiped on every restart.

## 1. One-time setup

1. Create a Neon project and copy its connection string
   (`postgresql://…?sslmode=require`). Tables are created on first start.
2. On huggingface.co create a Space: SDK **Docker**, template **Blank**.
3. In the Space's **Settings → Variables and secrets**, add:

| Name | Kind | Required | Notes |
| --- | --- | --- | --- |
| `JWT_SECRET` | secret | yes | `python -c "import secrets; print(secrets.token_hex(32))"`. Without it, sessions reset on every restart. |
| `DATABASE_URL` | secret | yes | Neon connection string. Without it, a SQLite file is used and wiped on restart. |
| `GROQ_API_KEY` | secret | optional | Without it, reports and chat use template text. |
| `GOOGLE_CLIENT_ID` | secret | optional | Google sign-in. Add `https://<space-host>/api/auth/google/callback` as an authorised redirect URI. |
| `GOOGLE_CLIENT_SECRET` | secret | optional | Needed together with the client ID. |
| `GMAIL_ADDRESS` | secret | optional | SMTP email. See the note below. |
| `GMAIL_APP_PASSWORD` | secret | optional | Needed together with the address. |
| `FRONTEND_URL`, `BACKEND_URL` | variable | optional | Defaults to `https://$SPACE_HOST` and `https://$SPACE_HOST/api`. Set them only for a custom domain. |
| `MODALITY_GATE_CONFIDENCE_THRESHOLD` | variable | optional | Default `0.90`. |

The Dockerfile already sets `SKIP_REPRO_CHECK=1` and `TRUSTED_PROXY_IPS`.

## 2. Deploy

```bash
brew install git-lfs                 # once
git commit …                         # only committed code is deployed
DRY_RUN=1 HF_USERNAME=me HF_SPACE_NAME=my-space scripts/deploy_hf.sh  # check size
HF_USERNAME=me HF_SPACE_NAME=my-space HF_TOKEN=hf_… scripts/deploy_hf.sh
```

`HF_TOKEN` is a **write** token from huggingface.co/settings/tokens. Each
push replaces the Space's history with one fresh commit, and the weights
upload again every time.

## 3. After the first start

- Check the Space logs for `Rate-limit key from proxy chain: peer=… x-forwarded-for=… -> …`.
  The address after `->` should be your own public IP. If it shows a proxy
  address instead, add that proxy's network to `TRUSTED_PROXY_IPS`.
- Use the direct URL `https://<user>-<space>.hf.space`, not the
  huggingface.co/spaces/… page. On that page the app runs in an iframe,
  where Google blocks its sign-in page and the OAuth nonce cookie counts as
  a third-party cookie.
- Free Spaces sleep after about 48 h without visitors. The first request
  afterwards waits for the container to start again.
