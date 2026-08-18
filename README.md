# Questify

Server-side extraction service for MockMate. A small FastAPI + PyMuPDF (fitz)
application that parses a PDF (or raster image), extracts questions and
answers via layout analysis, renders each question as a small PNG, and
returns a single JSON payload that the MockMate frontend (`PdfImportReview`)
consumes directly.

## Endpoints

| Method | Path       | Description                                                    |
|--------|------------|---------------------------------------------------------------|
| GET    | `/health`  | Liveness / wake-up probe. Returns `{"ok":true,...}`.          |
| POST   | `/extract` | `multipart/form-data`, field `file` = PDF or image. Returns the extracted questions as JSON. |

## Run locally

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Deploy on Render

`render.yaml` defines the service. Render auto-builds from the `requirements.txt`
and starts via `uvicorn app:app --host 0.0.0.0 --port $PORT`.

## Concurrency

`process_pdf` is CPU-bound and synchronous. It is offloaded to a thread via
`asyncio.to_thread` so uvicorn's event loop stays free to handle concurrent
requests from multiple users. Each call uses its own temp directory (no shared
state or file collisions).
