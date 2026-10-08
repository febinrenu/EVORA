"""Background pre-rendering of the top evidence thumbnails and clips, so playback is instant."""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor

from evora.core import cameras as cams
from evora.core.db import Database
from evora.core.media_service import MediaError, MediaService
from evora.evidence import store

log = logging.getLogger("evora.prerender")


class Prerenderer:
    def __init__(self, db: Database, media: MediaService, blur_enabled: Callable[[], bool], top: int = 3, workers: int = 2):
        self.db, self.media, self.blur_enabled, self.top = db, media, blur_enabled, top
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prerender")

    def schedule(self, evidence_ids: Iterable[str]) -> list[Future]:
        """Queue the first `top` ids. Never raises: a failed render is logged and the on-demand route still works."""
        return [self._pool.submit(self._render, eid) for eid in list(evidence_ids)[: self.top]]

    def _render(self, evidence_id: str) -> bool:
        try:
            rec = store.get(self.db, evidence_id)
            cam = cams.get_camera(self.db, rec.camera_id)
            want_blur = self.blur_enabled()
            self.media.thumb(cam, rec, want_blur)
            self.media.clip(cam, rec, want_blur)
        except (store.EvidenceError, store.EvidenceNotFound, cams.CameraNotFound, MediaError) as exc:
            log.warning("prerender skipped %s: %s", evidence_id, exc)
            return False
        return True

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
