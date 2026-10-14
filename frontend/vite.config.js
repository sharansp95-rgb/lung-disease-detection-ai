import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Prefix '' so non-VITE_ keys (never exposed to the bundle) load too.
  const env = loadEnv(mode, process.cwd(), '')

  // Where /api/* is forwarded. The backend stays bound to 127.0.0.1 and is
  // only reachable from outside through this proxy (docs/DEPLOYMENT_PUBLIC.md).
  const backendTarget = env.BACKEND_PROXY_TARGET || 'http://127.0.0.1:8000'

  // Extra Host headers to accept, e.g. the ngrok dev domain. Accepts bare
  // hostnames or full URLs, comma-separated. Empty = Vite's default
  // (localhost and IP addresses only), so plain local dev is unchanged.
  const allowedHosts = (env.FRONTEND_ALLOWED_HOSTS || '')
    .split(',')
    .map((h) => h.trim().replace(/^https?:\/\//, '').replace(/\/.*$/, ''))
    .filter(Boolean)

  const proxy = {
    '^/api(/|$)': {
      target: backendTarget,
      // Keep the original Host header (the public domain), not the target's.
      changeOrigin: false,
      rewrite: (path) => path.replace(/^\/api/, '') || '/',
      configure: (proxyServer) => {
        // Set forwarding headers by hand instead of `xfwd: true`, which
        // would append to X-Forwarded-Proto ("https,http") rather than keep
        // the value ngrok set. X-Forwarded-For gets this hop appended so
        // the backend can find the real visitor IP (see rate_limit.py).
        // Done on 'start' (before the outgoing headers are built) rather
        // than 'proxyReq', which the proxy skips for requests carrying an
        // `Expect: 100-continue` header (e.g. large multipart uploads).
        proxyServer.on('start', (req) => {
          const peer = req.socket.remoteAddress || ''
          const forwardedFor = req.headers['x-forwarded-for']
          req.headers['x-forwarded-for'] = forwardedFor ? `${forwardedFor}, ${peer}` : peer
          if (!req.headers['x-forwarded-proto']) {
            req.headers['x-forwarded-proto'] = req.socket.encrypted ? 'https' : 'http'
          }
          if (!req.headers['x-forwarded-host'] && req.headers.host) {
            req.headers['x-forwarded-host'] = req.headers.host
          }
        })
      },
    },
  }

  const hostOptions = {
    proxy,
    ...(allowedHosts.length ? { allowedHosts } : {}),
  }

  return {
    plugins: [react()],
    server: hostOptions,
    preview: hostOptions,
  }
})
