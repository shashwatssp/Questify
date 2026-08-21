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
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from queue import Empty, SimpleQueue
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from extractor import process_pdf

# ── Logging ─────────────────────────────────────────────────────
# Structured, timestamped logs to stdout/stderr so every extraction phase is
# observable in Render's log stream (the service previously had NO app-level
# logging, which is why large-PDF failures were impossible to diagnose).
logging.basicConfig(
    level=os.environ.get("QUESTIFY_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("questify")

# ── Timeouts ────────────────────────────────────────────────────
# Hard cap on how long a single extraction may run. Bounds CPU-bound parsing so
# the service never hangs silently on a huge PDF — it fails loudly with a
# progress-error frame instead. Override via the env var in seconds.
EXTRACTION_TIMEOUT = float(os.environ.get("EXTRACTION_TIMEOUT", "1800"))
# How often the SSE endpoint emits a heartbeat while the worker is alive but
# not producing events, so clients/proxies can detect a live-but-silent stream
# (preventing a connection being dropped mid-parse for large files).
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", "15"))

# --- Async job registry (POST /extract/start -> GET /extract/status/{job_id}) ---
# Decouples long extractions from a single long-lived HTTP connection: the file
# is saved and a background thread runs process_pdf while the client polls
# status by job_id. This defeats SSE proxy buffering, browser tab throttling and
# Render's connection limits -- the browser can close/reopen and resume polling
# the same job_id. The free tier (512MB) is memory-bounded by JOB_TTL.
JOBS: dict = {}
JOB_TTL = float(os.environ.get("JOB_TTL", "3600"))          # hard cap incl. running
JOB_DONE_TTL = float(os.environ.get("JOB_DONE_TTL", "300")) # reap terminal jobs after this
JOB_SWEEP_INTERVAL = float(os.environ.get("JOB_SWEEP_INTERVAL", "60"))
# Rough per-question seconds used to estimate total time before the client has
# a full answer count. Tuned for Render free-tier shared CPU.
ESTIMATE_SECS_PER_QUESTION = float(os.environ.get("ESTIMATE_SECS_PER_QUESTION", "5.0"))

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
    POST /extract/stream so both return identical payloads.

    Idempotent: questions emitted incrementally by process_pdf already carry
    their `rendered_image_b64` / figure `image_b64` (inlined at render time), so
    we only fill in anything still missing. This keeps the per-question
    streaming path and the final bundled payload consistent without re-encoding
    images that were already inlined.
    """
    data = result["data"]
    img_dir = result["img_dir"]
    for q in data.get("questions", []):
        if q.get("rendered_image") and not q.get("rendered_image_b64"):
            q["rendered_image_b64"] = _to_data_url(img_dir, q["rendered_image"])
        for fig in q.get("figures", []):
            if fig.get("path") and not fig.get("image_b64"):
                fig["image_b64"] = _to_data_url(img_dir, fig["path"])
    return data


def _process_pdf_sync(tmp_path: str, timeout: float | None = None) -> dict:
    """Run process_pdf + inline images as base64.

    Extracted into a standalone sync function so the async /extract endpoint
    can offload it to a thread via asyncio.to_thread, which keeps uvicorn's
    event loop free to handle concurrent requests from multiple users.
    Each call gets its own temp output dir (mkdtemp inside process_pdf),
    so there are no shared-file collisions across concurrent invocations.
    `timeout` is forwarded to process_pdf's internal deadline checks so a runaway
    extraction raises TimeoutError instead of hanging forever.
    """
    result = process_pdf(tmp_path, timeout=timeout)
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
        shutil.rmtree(upload_dir, ignore_errors=True)
        logger.warning("extract: rejected unsupported upload name=%s bytes=%d", file.filename, len(content))
        raise HTTPException(400, "Only PDF or image files are supported.")
    tmp_path = os.path.join(upload_dir, base if kind == "pdf" else f"upload.{kind}")
    with open(tmp_path, "wb") as f:
        f.write(content)
    logger.info("extract: START kind=%s bytes=%d name=%s", kind, len(content), base)
    result: Optional[dict] = None
    try:
        if kind == "pdf":
            # Run in a worker thread with BOTH an internal deadline (process_pdf
            # raises TimeoutError so cleanup runs) and an outer asyncio timeout as a
            # hard cap for any single call that can't be interrupted. A bounded
            # timeout is what turns a silent hang into a loud, logged 504.
            result = await asyncio.wait_for(
                asyncio.to_thread(_process_pdf_sync, tmp_path, EXTRACTION_TIMEOUT),
                timeout=EXTRACTION_TIMEOUT + 5,
            )
            data = result["data"]
            logger.info("extract: DONE kind=pdf questions=%d", len(data.get("questions", [])))
        else:
            # Raster image: no selectable text layer. Return it as a single
            # image-backed question so it still enters the review flow, where the
            # teacher can crop and run on-demand OCR per question.
            data = await asyncio.to_thread(_process_image_sync, content, base)
            logger.info("extract: DONE kind=%s", kind)
        return JSONResponse(content=data)
    except HTTPException:
        raise
    except TimeoutError as e:
        logger.error("extract: TIMED OUT kind=%s name=%s error=%s", kind, base, e)
        raise HTTPException(
            504,
            f"Extraction timed out after {EXTRACTION_TIMEOUT:.0f}s. The PDF/image may be "
            "too large or complex; try a smaller file or fewer pages.",
        ) from e
    except Exception as e:  # surface extraction failures as a clean HTTP error
        logger.exception("extract: FAILED kind=%s name=%s", kind, base)
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

    The worker runs in a background thread bounded by `EXTRACTION_TIMEOUT` (an
    internal deadline inside process_pdf turns a runaway parse into a
    `progress-error` frame instead of a silent, forever-hung stream). While
    the worker is alive but quiet a `progress-heartbeat` frame is emitted every
    `HEARTBEAT_INTERVAL` seconds so clients and proxies can tell the request is
    still alive (this is what prevents large files from looking like they "never
    start parsing").
    """
    start = time.monotonic()
    upload_dir = tempfile.mkdtemp(prefix="questify_upload_")
    base = (file.filename or "upload")
    for c in ("/", " ", chr(92)):
        base = base.replace(c, "_")
    content = await file.read()
    kind = _media_kind(content)
    if not kind:
        shutil.rmtree(upload_dir, ignore_errors=True)
        logger.warning("stream: rejected unsupported upload name=%s bytes=%d", file.filename, len(content))
        raise HTTPException(400, "Only PDF or image files are supported.")
    tmp_path = os.path.join(upload_dir, base if kind == "pdf" else f"upload.{kind}")
    with open(tmp_path, "wb") as f:
        f.write(content)

    logger.info("stream: START kind=%s bytes=%d name=%s timeout=%ss", kind, len(content), base, EXTRACTION_TIMEOUT)

    q: SimpleQueue = SimpleQueue()
    worker_done = threading.Event()

    def on_progress(event: dict) -> None:
        q.put(event)

    def worker() -> None:
        result = None
        try:
            if kind == "pdf":
                result = process_pdf(tmp_path, on_progress=on_progress,
                                     timeout=EXTRACTION_TIMEOUT)
                data = _inline_images(result)
                logger.info("stream: process_pdf completed in %.2fs questions=%d",
                            time.monotonic() - start, len(data.get("questions", [])))
            else:
                # Raster image: single image-backed question (no text layer).
                data = _process_image_sync(content, base)
                q.put({"event": "progress-start",
                       "data": {"total_pages": 1, "total_questions": 1}})
                q.put({"event": "progress-question",
                       "data": {"index": 1, "count": 1, "total": 1, "page": 1}})
            q.put({"event": "progress-done", "data": data})
        except TimeoutError as e:
            logger.error("stream: TIMED OUT name=%s elapsed=%.1fs error=%s", base, time.monotonic() - start, e)
            q.put({"event": "progress-error", "data": {"error": str(e)}})
        except Exception as e:
            logger.exception("stream: extraction FAILED name=%s", base)
            q.put({"event": "progress-error", "data": {"error": str(e)}})
        finally:
            worker_done.set()
            q.put(None)  # sentinel: close the stream
            shutil.rmtree(upload_dir, ignore_errors=True)
            if result is not None:
                shutil.rmtree(os.path.dirname(result["img_dir"]), ignore_errors=True)

    threading.Thread(target=worker, daemon=True).start()

    # Block briefly on each get so a long-silent worker (common for large PDFs
    # stuck in detection/rendering) still produces keep-alive frames instead of
    # letting an idle proxy tear the connection down → silent failure.
    _TIMEOUT = object()

    def _safe_get(block: bool, timeout: float):
        try:
            return q.get(block, timeout)
        except Empty:
            return _TIMEOUT

    async def event_source():
        while True:
            item = await asyncio.to_thread(_safe_get, True, HEARTBEAT_INTERVAL)
            if item is _TIMEOUT:
                if worker_done.is_set():
                    return
                # Worker alive but quiet — emit a heartbeat so the client/proxy
                # knows the stream is still in progress.
                yield ("event: progress-heartbeat\ndata: "
                       + json.dumps({"elapsed": round(time.monotonic() - start, 1)})
                       + "\n\n")
                continue
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


# ── Async job model (POST /extract/start -> GET /extract/status/{job_id}) ──
def _init_job_state() -> dict:
    """Fresh state dict for a new extraction job."""
    now = time.monotonic()
    return {
        "status": "running",
        "processed": 0,
        "total": None,
        "total_pages": None,
        "error": None,
        "data": None,
        # Accumulated, fully-inlined questions produced so far. Delivered to
        # clients on every status poll so the teacher can start reviewing the
        # first questions while the bulk of the PDF is still extracting.
        "results": [],
        "created_at": now,
        "updated_at": now,
        "done_at": None,
    }


def _job_on_progress(state: dict, event: dict) -> None:
    """Translate process_pdf progress events into JOBS state fields."""
    ev = event.get("event")
    d = event.get("data") or {}
    if ev == "progress-start":
        if d.get("total_questions") is not None:
            state["total"] = d["total_questions"]
        if d.get("total_pages") is not None:
            state["total_pages"] = d["total_pages"]
    elif ev == "progress-question":
        state["processed"] = d.get("count", d.get("index", 0))
        if d.get("total") is not None:
            state["total"] = d["total"]
        # Attach the fully-inlined question (image + figures as base64) so the
        # polling client can surface it immediately for incremental review.
        q = d.get("question")
        if q is not None:
            state["results"].append(q)
    state["updated_at"] = time.monotonic()


def _run_job(job_id: str, kind: str, tmp_path: str, base: str,
             content: bytes, upload_dir: str) -> None:
    """Background worker: runs process_pdf() (same logic as the SSE worker)
    and records progress + the final result into JOBS[job_id]."""
    state = JOBS[job_id]
    result = None
    try:
        if kind == "pdf":
            logger.info("job: worker running job=%s path=%s", job_id, tmp_path)
            result = process_pdf(
                tmp_path,
                on_progress=lambda ev: _job_on_progress(state, ev),
                timeout=EXTRACTION_TIMEOUT,
            )
            data = _inline_images(result)
            state["processed"] = len(data.get("questions", []))
            state["total"] = len(data.get("questions", []))
            state["status"] = "done"
            state["data"] = data
            logger.info("job: DONE job=%s questions=%d", job_id, len(data.get("questions", [])))
        else:
            data = _process_image_sync(content, base)
            state["processed"] = 1
            state["total"] = 1
            state["total_pages"] = 1
            state["status"] = "done"
            state["data"] = data
    except TimeoutError as e:
        logger.error("job: TIMED OUT job=%s error=%s", job_id, e)
        state["status"] = "error"
        state["error"] = str(e)
    except Exception as e:
        logger.exception("job: FAILED job=%s", job_id)
        state["status"] = "error"
        state["error"] = str(e)
    finally:
        state["done_at"] = time.monotonic()
        state["updated_at"] = time.monotonic()
        # Images are already inlined as base64 in state["data"]; release the
        # per-question scratch dir now so memory is freed before the sweep.
        if result is not None:
            shutil.rmtree(os.path.dirname(result["img_dir"]), ignore_errors=True)
        shutil.rmtree(upload_dir, ignore_errors=True)


def _jobs_sweep():
    """Background reaper: drop terminal jobs after JOB_DONE_TTL and any job
    older than JOB_TTL. Bounds memory on the 512MB free tier."""
    while True:
        time.sleep(JOB_SWEEP_INTERVAL)
        now = time.monotonic()
        for jid in list(JOBS.keys()):
            st = JOBS[jid]
            age = now - st["created_at"]
            if st["status"] == "running":
                if age > JOB_TTL:
                    logger.warning("job: sweep expired running job=%s age=%.0fs", jid, age)
                    JOBS.pop(jid, None)
            else:
                done_at = st.get("done_at") or st["updated_at"]
                if (now - done_at) > JOB_DONE_TTL or age > JOB_TTL:
                    logger.info("job: sweep reaped terminal job=%s status=%s", jid, st["status"])
                    JOBS.pop(jid, None)


@app.post("/extract/start")
async def extract_start(file: UploadFile = File(...)):
    """Start an async extraction job.

    Saves the upload to disk, immediately returns {"job_id": "..."} and runs
    process_pdf() in a background thread (same logic as /extract/stream, just
    writing progress into JOBS instead of an SSE stream). Poll
    GET /extract/status/{job_id} for progress and the final result.

    Recommended for large PDFs (100-300+ questions) on the free tier: no
    long-lived HTTP connection to drop, so the browser can be closed/reopened
    and resume polling the same job_id. POST /extract and POST /extract/stream
    remain for backward compatibility.
    """
    upload_dir = tempfile.mkdtemp(prefix="questify_upload_")
    base = (file.filename or "upload").replace("/", "_").replace("\\", "_")
    content = await file.read()
    kind = _media_kind(content)
    if not kind:
        shutil.rmtree(upload_dir, ignore_errors=True)
        logger.warning("start: rejected unsupported upload name=%s bytes=%d",
                       file.filename, len(content))
        raise HTTPException(400, "Only PDF or image files are supported.")
    tmp_path = os.path.join(upload_dir, base if kind == "pdf" else f"upload.{kind}")
    with open(tmp_path, "wb") as f:
        f.write(content)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = _init_job_state()
    logger.info("start: START job=%s kind=%s bytes=%d", job_id, kind, len(content))
    threading.Thread(
        target=_run_job,
        args=(job_id, kind, tmp_path, base, content, upload_dir),
        daemon=True,
        name=f"questify-job-{job_id[:8]}",
    ).start()
    return JSONResponse(content={"job_id": job_id, "status": "running"})


@app.get("/extract/status/{job_id}")
def extract_status(job_id: str):
    """Poll an extraction job.

    Returns progress (processed/total) while running, and the full `data`
    payload once status == "done". 404 if missing/expired (reaped
    JOB_DONE_TTL seconds after a terminal state, or JOB_TTL hard cap).
    """
    state = JOBS.get(job_id)
    if state is None:
        raise HTTPException(
            404,
            "Job not found. It may have expired after completion "
            f"({int(JOB_DONE_TTL)}s after a terminal state, {int(JOB_TTL)}s hard cap).",
        )
    total = state["total"]
    out = {
        "job_id": job_id,
        "status": state["status"],
        "processed": state["processed"],
        "total": total,
        "total_pages": state.get("total_pages"),
        "error": state.get("error"),
        "estimated_seconds": total * ESTIMATE_SECS_PER_QUESTION if total else None,
        # Fully-inlined questions produced so far. Always present once extraction
        # has started emitting questions (additive field -> old clients that
        # don't read it are unaffected).
        "results": state.get("results", []),
    }
    if state["status"] == "done" and state["data"] is not None:
        out["data"] = state["data"]
    return JSONResponse(content=out)


# Start the job-reaper sweep once at import time (Render runs a single uvicorn
# worker, so one reaper thread covers all jobs in this process).
threading.Thread(target=_jobs_sweep, daemon=True, name="questify-job-sweep").start()

