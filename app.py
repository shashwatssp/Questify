"""
Questify — server-side PDF question extraction service.

FastAPI wrapper around the PyMuPDF extractor (`extractor.process_pdf`).  The
MockMate web frontend POSTs a PDF *or image* here and receives structured
questions with rendered images inlined as base64 data-URLs, so no second
round-trip is needed. Images (no text layer) come back as a single
image-backed question; the review screen can OCR it on demand.

No LLM, no credits — Questify reads the PDF text layer (or hands back the
rendered image for on-demand review-screen OCR).

Endpoints
---------
GET  /health   — liveness probe. The frontend pings this on load to wake a
                free-tier Render container that would otherwise sleep after a
                few minutes of inactivity (Render's "always free" web service
                spins down; a hit keeps it warm).
POST /extract  — multipart/form-data, field `file` = PDF or image.
                Returns the `data` dict (questions + metadata) with each
                question's rendered image and figures as base64 data-URLs.
GET  /docs     — FastAPI's auto-generated OpenAPI UI.
"""

import asyncio
import base64
import io
import json
import os
import shutil
import tempfile
import threading
from queue import SimpleQueue
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from extractor import process_pdf

app = FastAPI(title="Questify", version="0.1.0")

# Local dev: Vite on :5173 calls this service. In production the Netlify-origin
# value would be pinned here; "*" is deliberate for local validation.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_data_url(img_dir: str, fname: Optional[str]) -> Optional[str]:
    """Read an image file from `img_dir` and return it as a base64 data-URL."""
    if not fname:
        return None
    path = os.path.join(img_dir, fname)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _media_kind(content: bytes) -> str:
    """Classify an upload by magic bytes.

    Returns 'pdf', 'png', 'jpg', 'gif', 'webp' or ''. Validation is done by
    content, not by extension, so a valid PDF named ``pdf.neet`` is accepted.
    """
    if content[:5] == b"%PDF-":
        return "pdf"
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if content[:3] == b"\xff\xd8\xff":
        return "jpg"
    if content[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if content[:12] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp"
    return ""


def _image_bytes_to_data_url(content: bytes, max_dim: int = 1800) -> str:
    """Render raster image bytes to a base64 PNG data-URL (~2x, size-capped).

    No OCR: a raster image has no text layer, so it is returned image-backed
    and the MockMate review screen parses it on demand with its existing
    tesseract.js toggle (no LLM, no credits). The cap keeps photos from
    blowing up memory/decoding time.
    """
    im = Image.open(io.BytesIO(content))
    if im.mode not in ("RGB", "L", "P"):
        im = im.convert("RGB")
    w, h = im.size
    scale = min(2.0, max_dim / max(w, h))
    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _image_backed_question(b64: str, source_name: str) -> dict:
    """A single image-backed question wrapping a raster image (no text layer)."""
    return {
        "exam": "", "subject": "", "topic": "", "date": "",
        "total_marks": "", "duration": "",
        "source_pdf": source_name, "total_questions": 1,
        "questions": [{
            "id": 1, "text": "", "type": "image", "options": {},
            "equations": [], "figures": [],
            "rendered_image": "", "rendered_image_b64": b64,
            "raw_text": "", "page": 1,
        }],
    }


def _inline_images(result: dict) -> dict:
    """Inline each question's rendered image + figures as base64 data-URLs
    so the frontend needs no second request. Shared by POST /extract and
    POST /extract/stream so both return identical payloads."""
    data = result["data"]
    img_dir = result["img_dir"]
    for q in data.get("questions", []):
        if q.get("rendered_image"):
            q["rendered_image_b64"] = _to_data_url(img_dir, q["rendered_image"])
        for fig in q.get("figures", []):
            if fig.get("path"):
                fig["image_b64"] = _to_data_url(img_dir, fig["path"])
    return data


def _process_pdf_sync(tmp_path: str) -> dict:
    """Run process_pdf + inline images as base64.

    Extracted into a standalone sync function so the async /extract endpoint
    can offload it to a thread via asyncio.to_thread, which keeps uvicorn's
    event loop free to handle concurrent requests from multiple users.
    Each call gets its own temp output dir (mkdtemp inside process_pdf),
    so there are no shared-file collisions across concurrent invocations.
    """
    result = process_pdf(tmp_path)
    _inline_images(result)  # mutates result["data"] in place; keep full dict for cleanup
    return result


def _process_image_sync(content: bytes, source_name: str) -> dict:
    """Render a raster image to a base64 data-URL (runs in a thread)."""
    return _image_backed_question(_image_bytes_to_data_url(content), source_name)


@app.get("/health")
def health() -> dict:
    """Liveness / warm-up probe. Returns ok so a caller can wake a sleeping
    container (Render free tier sleeps after ~15 min of inactivity)."""
    return {"ok": True, "service": "questify"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Extract questions from a PDF (via extractor.process_pdf) or a raster image.

    Images have no text layer, so they are returned as a single image-backed
    question; the MockMate review screen can OCR them on demand — no LLM, no credits.
    """
    upload_dir = tempfile.mkdtemp(prefix="questify_upload_")
    base = (file.filename or "upload").replace("/", "_").replace("\\", "_")
    content = await file.read()
    kind = _media_kind(content)
    if not kind:
        raise HTTPException(400, "Only PDF or image files are supported.")
    tmp_path = os.path.join(upload_dir, base if kind == "pdf" else f"upload.{kind}")
    with open(tmp_path, "wb") as f:
        f.write(content)
    result: Optional[dict] = None
    try:
        if kind == "pdf":
            result = await asyncio.to_thread(_process_pdf_sync, tmp_path)
            data = result["data"]
        else:
            # Raster image: no selectable text layer. Return it as a single
            # image-backed question so it still enters the review flow, where the
            # teacher can crop and run on-demand OCR per question.
            data = await asyncio.to_thread(_process_image_sync, content, base)
        return JSONResponse(content=data)
    except HTTPException:
        raise
    except Exception as e:  # surface extraction failures as a clean HTTP error
        raise HTTPException(500, f"Extraction failed: {e}") from e
    finally:
        # Clean up this request's scratch files (upload dir + process_pdf output).
        shutil.rmtree(upload_dir, ignore_errors=True)
        if result is not None:
            shutil.rmtree(os.path.dirname(result["img_dir"]), ignore_errors=True)


@app.post("/extract/stream")
async def extract_stream(file: UploadFile = File(...)):
    """Stream extraction progress as Server-Sent Events.

    POST is required because the upload body cannot be sent via a GET
    (the browser EventSource API only issues GETs); the response body still
    uses the standard SSE wire format so the frontend parses it with the
    Streams API. Emits, in order: progress-start, one progress-question per
    extracted question, then progress-done (or progress-error). POST /extract
    remains the non-streaming fallback and returns an identical payload.
    """
    upload_dir = tempfile.mkdtemp(prefix="questify_upload_")
    base = (file.filename or "upload")
    for c in ("/", " ", chr(92)):
        base = base.replace(c, "_")
    content = await file.read()
    kind = _media_kind(content)
    if not kind:
        raise HTTPException(400, "Only PDF or image files are supported.")
    tmp_path = os.path.join(upload_dir, base if kind == "pdf" else f"upload.{kind}")
    with open(tmp_path, "wb") as f:
        f.write(content)

    q: SimpleQueue = SimpleQueue()

    def on_progress(event: dict) -> None:
        q.put(event)

    def worker() -> None:
        result = None
        try:
            if kind == "pdf":
                result = process_pdf(tmp_path, on_progress=on_progress)
                data = _inline_images(result)
            else:
                # Raster image: single image-backed question (no text layer).
                data = _process_image_sync(content, base)
                q.put({"event": "progress-start",
                       "data": {"total_pages": 1, "total_questions": 1}})
                q.put({"event": "progress-question",
                       "data": {"index": 1, "count": 1, "total": 1, "page": 1}})
            q.put({"event": "progress-done", "data": data})
        except Exception as e:
            q.put({"event": "progress-error", "data": {"error": str(e)}})
        finally:
            q.put(None)  # sentinel: close the stream
            shutil.rmtree(upload_dir, ignore_errors=True)
            if result is not None:
                shutil.rmtree(os.path.dirname(result["img_dir"]), ignore_errors=True)

    threading.Thread(target=worker, daemon=True).start()

    async def event_source():
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                return
            payload = json.dumps(item["data"])
            yield f"event: {item['event']}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
