"""Serve the built single-page UI from the same origin as the API."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class SPAStaticFiles(StaticFiles):
    """Static files where unknown paths fall back to index.html (client-side routes), except under /api."""

    async def get_response(self, path: str, scope: Scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            if path.replace("\\", "/").split("/", 1)[0] == "api":  # Windows hands the path over with backslashes
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            return await super().get_response("index.html", scope)


def mount_ui(app: FastAPI, ui_dir: Path) -> bool:
    """Mount `ui_dir` at `/` after every API route. Returns False when there is no built UI."""
    if not (ui_dir / "index.html").is_file():
        return False
    app.mount("/", SPAStaticFiles(directory=str(ui_dir), html=True), name="ui")
    return True
