# Deploying to one AWS EC2 instance

The whole app runs on one Ubuntu server with Docker Compose
(`docker-compose.yml`):

- **app**: the image from `Dockerfile`. It serves the React build at `/` and the API under `/api`, on port 7860, reachable only inside Docker.
- **caddy**: listens on ports 80 and 443 and gets a free Let's Encrypt certificate for your domain automatically.

User accounts and scan history are stored in SQLite on a Docker volume, so
they survive rebuilds and redeploys.

The domain is a free [sslip.io](https://sslip.io) name. `13-233-1-2.sslip.io`
always resolves to `13.233.1.2`, so there's nothing to register.

## 1. AWS console (by hand)

Switch the region (top right) to **Asia Pacific (Mumbai) ap-south-1**.

1. **Key pair**: EC2 → Key pairs → *Create key pair*. Choose ED25519 and
   the `.pem` format. Save it as `~/.ssh/lung-app.pem`, then run
   `chmod 400 ~/.ssh/lung-app.pem`.
2. **Security group**: EC2 → Security groups → *Create*. Add these inbound
   rules and nothing else:

   | Type | Port | Source |
   | --- | --- | --- |
   | SSH | 22 | *My IP* |
   | HTTP | 80 | Anywhere-IPv4 (0.0.0.0/0) |
   | HTTPS | 443 | Anywhere-IPv4 (0.0.0.0/0) |

   Port 80 is required: Let's Encrypt validates over it, and Caddy uses it
   to redirect visitors to HTTPS. If your home IP changes, update the SSH
   rule.
3. **Launch the instance**: EC2 → *Launch instance*.
   - **AMI:** Ubuntu Server 24.04 LTS, 64-bit (x86).
   - **Instance type:** `c7i-flex.large` or `m7i-flex.large`.
   - **Key pair and security group:** the two from steps 1 and 2.
   - **Storage:** 30 GiB gp3. The default 8 GiB is too small for the image and build cache.
4. **Elastic IP**: EC2 → Elastic IPs → *Allocate*, then *Associate* it with
   the instance. This keeps the IP, and with it your sslip.io domain, across
   stop and start. Note that AWS bills public IPv4 addresses, including
   Elastic IPs, beyond any free-tier allowance.
5. **Your domain** is the Elastic IP with dots replaced by dashes, plus
   `.sslip.io`. For example, `13.233.1.2` becomes `13-233-1-2.sslip.io`.
6. **Google sign-in** (optional): in Google Cloud Console → APIs & Services
   → Credentials → your OAuth client, add this authorised redirect URI:
   `https://<domain>/api/auth/google/callback`. While the consent screen is
   in *Testing* mode, only the test users you list there can sign in.

## 2. Prepare the server (once)

```bash
ssh -i ~/.ssh/lung-app.pem ubuntu@<ELASTIC_IP> 'sudo bash -s' < scripts/server_setup.sh
```

This installs Docker and the compose plugin, adds a 4 GB swap file, enables
unattended security upgrades and creates `/opt/lung-app`. It is safe to
re-run.

It also clamps the TCP segment size (MSS) the server advertises to 1220.
On some mobile networks (Jio was measured: 1260 works, 1300 fails),
full-size packets from the client are silently dropped. Without the clamp,
uploads from those networks stall and get reset, while downloads work. That
covers SSH and rsync (the deploy itself) as well as scan uploads over HTTPS.

## 3. Deploy (and every redeploy)

Commit first: only committed code is deployed.

```bash
SERVER_IP=<ELASTIC_IP> SSH_KEY=~/.ssh/lung-app.pem DOMAIN=<domain> \
  COPY_SECRETS=1 scripts/deploy_ec2.sh
```

- **First run:** creates `/opt/lung-app/.env.production` on the server with
  a generated `JWT_SECRET`. With `COPY_SECRETS=1` it also copies the Groq,
  Google and Gmail values from your local `backend/.env`; without it, those
  stay blank for you to fill in.
- **Every run:** uploads the code and weights, rebuilds, restarts, then
  waits for `https://<domain>/api/health` and prints the result.
- **First build:** takes roughly 5–15 minutes. Later builds reuse cached
  layers unless `requirements.txt` changes.
- **Re-running:** an existing `.env.production` is never replaced, and the
  database, logs and certificates are kept.

## 4. Day to day (on the server)

```bash
cd /opt/lung-app
sudo docker compose ps
sudo docker compose logs -f app            # app log; emails print here if SMTP isn't set
nano .env.production                       # then apply with:
sudo docker compose up -d
# Back up the SQLite database:
sudo docker compose exec -T app python -c "import sqlite3; sqlite3.connect('/data/app.db').backup(sqlite3.connect('/data/backup.db'))"
sudo docker compose cp app:/data/backup.db ./app-backup-$(date +%F).db
```

Never run `docker compose down -v`: the `-v` deletes the volumes, which
hold the database and the certificates.

## Environment (`/opt/lung-app/.env.production`)

| Name | Required | Notes |
| --- | --- | --- |
| `DOMAIN` | yes | Written by the deploy script on every run. |
| `JWT_SECRET` | yes | Generated on the first deploy. Changing it signs everyone out. |
| `GROQ_API_KEY` | no | Empty means reports and chat use template text. |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | no | Empty means Google sign-in shows "not configured". |
| `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` | no | Gmail app password, SMTP on port 587. Empty means links are printed to the app log. |
| `DATABASE_URL` | no | Postgres instead of SQLite. |

`docker-compose.yml` sets the URLs, `DB_PATH` and `TRUSTED_PROXY_IPS` (only
the Caddy container) itself.
