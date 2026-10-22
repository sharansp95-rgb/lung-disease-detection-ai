"""
Single-origin app for hosted deploys (Hugging Face Space, see Dockerfile):
the built frontend (frontend/dist) at "/" and the FastAPI app from main.py
under config.PROXY_API_PREFIX ("/api") -- the same layout the Vite /api
proxy gives in the ngrok setup, so the frontend is built with
VITE_API_BASE_URL=/api and the browser only ever sees one origin (no CORS).

    uvicorn serve:app --host 0.0.0.0 --port 7860

API requests keep their full path and get root_path="/api" (the ASGI
convention Starlette's own Mount uses), so routes match as before and
request URLs still show the public /api path. Lifespan events go to the
API app, so its startup (DB init + model preload) runs unchanged.

Any non-API path that isn't a file in dist/ and has no file extension gets
index.html, so client routes (/login, /results, /reset-password/<token>,
/oauth-callback, ...) survive a reload or a link from an email. Missing
assets (/assets/x.js) still 404.

Local dev is unaffected: `uvicorn main:app` + `npm run dev` never import this.
"""

import os
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.responses import PlainTextResponse
from starlette.staticfiles import StaticFiles

import config
from main import app as api_app

FRONTEND_DIST = Path(
    os.environ.get("FRONTEND_DIST", Path(__file__).resolve().parent.parent / "frontend" / "dist")
)


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            # Not mounted inside a Starlette app, so nothing else would turn
            # this into a response.
            if exc.status_code != 404 or "." in path.rsplit("/", 1)[-1]:
                return PlainTextResponse(exc.detail, status_code=exc.status_code, headers=exc.headers)
            return await super().get_response("index.html", scope)


def create_app(dist_dir: Path = FRONTEND_DIST, api=api_app, prefix: str = config.PROXY_API_PREFIX):
    if not (dist_dir / "index.html").is_file():
        raise RuntimeError(f"Frontend build not found at {dist_dir} -- run `npm run build` in frontend/.")
    frontend = SPAStaticFiles(directory=dist_dir, html=True)

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            await api(scope, receive, send)
            return
        path = scope["path"]
        if path == prefix or path.startswith(prefix + "/"):
            scope = dict(scope, root_path=scope.get("root_path", "") + prefix)
            await api(scope, receive, send)
            return
        await frontend(scope, receive, send)

    return app


app = create_app()
