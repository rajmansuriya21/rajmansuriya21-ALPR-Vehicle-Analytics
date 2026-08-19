"""
FastAPI web server for the Vehicle Analytics Dashboard.

Provides:
- REST API endpoints for configuration, processing, and data retrieval
- WebSocket endpoint for real-time processing updates
- Static file serving for the web dashboard

Usage:
    python -m app.server
    # or: uvicorn app.server:app --reload
"""

import asyncio
import base64
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.settings import load_settings
from core.pipeline import VideoPipeline

logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────
app = FastAPI(
    title="Vehicle Entry & Exit Analytics System",
    description="AI-powered CCTV vehicle analytics with license plate recognition",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Global State ──────────────────────────────────────────
settings = load_settings()
pipeline: Optional[VideoPipeline] = None
processing_thread: Optional[threading.Thread] = None
ws_clients: list = []
last_results: Optional[dict] = None


# ── WebSocket Manager ─────────────────────────────────────
async def broadcast_ws(data: dict):
    """Broadcast a message to all connected WebSocket clients."""
    message = json.dumps(data)
    disconnected = []
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        ws_clients.remove(ws)


# ── Routes ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML page."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not found. Check app/static/index.html</h1>")


@app.get("/api/status")
async def get_status():
    """Get system status."""
    global pipeline
    return {
        "status": "processing" if (pipeline and pipeline.is_running) else "idle",
        "config": {
            "video_source": settings.video_source,
            "line_mode": settings.line_mode,
            "camera_id": settings.camera_id,
            "device": settings.device,
        },
        "has_results": last_results is not None,
    }


@app.post("/api/process")
async def start_processing():
    """Start video processing using .env configuration."""
    global pipeline, processing_thread, last_results

    if pipeline and pipeline.is_running:
        return JSONResponse(
            status_code=409,
            content={"error": "Processing already in progress"},
        )

    try:
        video_paths = settings.get_video_paths()
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    def on_frame(frame, frame_idx, total_frames):
        """Called from processing thread with annotated frames."""
        try:
            # Encode frame as JPEG for WebSocket streaming
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            # Store for polling endpoint
            app.state.latest_frame = frame_b64
            app.state.progress = {
                "current_frame": frame_idx,
                "total_frames": total_frames,
                "percent": round((frame_idx / total_frames) * 100, 1),
            }
        except Exception:
            pass

    def process_worker():
        """Background processing thread."""
        global pipeline, last_results
        try:
            pipeline = VideoPipeline(
                settings=settings,
                on_frame=on_frame,
            )

            for video_path in video_paths:
                results = pipeline.process_video(video_path)
                last_results = results

            app.state.processing_complete = True
        except Exception as e:
            logger.error(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            app.state.processing_error = str(e)

    # Initialize state
    app.state.latest_frame = None
    app.state.progress = {"current_frame": 0, "total_frames": 0, "percent": 0}
    app.state.processing_complete = False
    app.state.processing_error = None

    # Start processing in background thread
    processing_thread = threading.Thread(target=process_worker, daemon=True)
    processing_thread.start()

    return {"message": "Processing started", "videos": video_paths}


@app.post("/api/stop")
async def stop_processing():
    """Stop ongoing video processing."""
    global pipeline
    if pipeline and pipeline.is_running:
        pipeline.stop()
        return {"message": "Stop signal sent"}
    return {"message": "No processing in progress"}


@app.get("/api/progress")
async def get_progress():
    """Get current processing progress."""
    progress = getattr(app.state, "progress", {})
    is_complete = getattr(app.state, "processing_complete", False)
    error = getattr(app.state, "processing_error", None)
    return {
        "progress": progress,
        "is_complete": is_complete,
        "error": error,
    }


@app.get("/api/frame")
async def get_latest_frame():
    """Get the latest annotated frame (base64 JPEG)."""
    frame = getattr(app.state, "latest_frame", None)
    if frame:
        return {"frame": frame}
    return {"frame": None}


@app.get("/api/events")
async def get_events():
    """Get all detected events."""
    global last_results
    if last_results:
        return {"events": last_results.get("events", [])}
    return {"events": []}


@app.get("/api/visits")
async def get_visits():
    """Get all visit records."""
    global last_results
    if last_results:
        return {"visits": last_results.get("visits", [])}
    return {"visits": []}


@app.get("/api/summary")
async def get_summary():
    """Get summary statistics."""
    global last_results
    if last_results:
        return {"summary": last_results.get("summary", {})}
    return {"summary": {}}


@app.get("/api/analytics")
async def get_analytics():
    """Get AI analytics report content."""
    global last_results
    if last_results:
        report_path = last_results.get("output_paths", {}).get("ai_report")
        if report_path and Path(report_path).exists():
            content = Path(report_path).read_text(encoding="utf-8")
            return {"report": content}
    return {"report": None}


@app.get("/api/download/{file_type}")
async def download_file(file_type: str):
    """Download generated output files."""
    global last_results
    if not last_results:
        return JSONResponse(status_code=404, content={"error": "No results available"})

    paths = last_results.get("output_paths", {})
    path_map = {
        "video": paths.get("annotated_video"),
        "json": paths.get("json_log"),
        "csv": paths.get("csv_report"),
        "analytics": paths.get("ai_report"),
    }

    file_path = path_map.get(file_type)
    if file_path and Path(file_path).exists():
        return FileResponse(
            file_path,
            filename=Path(file_path).name,
            media_type="application/octet-stream",
        )

    return JSONResponse(status_code=404, content={"error": f"File not found: {file_type}"})


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    upload_dir = Path("test")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Update settings to use uploaded video
    settings.video_source = str(file_path)

    return {"message": f"Video uploaded: {file.filename}", "path": str(file_path)}


@app.post("/api/configure-lines")
async def configure_lines(config: dict):
    """Update line coordinates via API."""
    if "entry_line" in config:
        settings.entry_line = config["entry_line"]
    if "exit_line" in config:
        settings.exit_line = config["exit_line"]
    if "single_line" in config:
        settings.single_line = config["single_line"]
    if "line_mode" in config:
        settings.line_mode = config["line_mode"]
    if "entry_direction" in config:
        settings.entry_direction = config["entry_direction"]

    return {"message": "Line configuration updated", "config": config}


# ── WebSocket ─────────────────────────────────────────────

@app.websocket("/ws/processing")
async def websocket_processing(websocket: WebSocket):
    """WebSocket endpoint for real-time processing updates."""
    await websocket.accept()
    ws_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(ws_clients)}")

    try:
        while True:
            # Send progress updates periodically
            progress = getattr(app.state, "progress", {})
            is_complete = getattr(app.state, "processing_complete", False)
            frame = getattr(app.state, "latest_frame", None)

            update = {
                "type": "progress",
                "data": progress,
                "is_complete": is_complete,
            }

            if frame:
                update["frame"] = frame

            await websocket.send_json(update)

            if is_complete:
                # Send final results
                global last_results
                if last_results:
                    await websocket.send_json({
                        "type": "complete",
                        "data": {
                            "events": last_results.get("events", []),
                            "visits": last_results.get("visits", []),
                            "summary": last_results.get("summary", {}),
                        },
                    })
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


# ── Main ──────────────────────────────────────────────────

def start_server():
    """Start the FastAPI server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("paddleocr").setLevel(logging.WARNING)
    logging.getLogger("ppocr").setLevel(logging.WARNING)

    print("\n" + "=" * 60)
    print("  🚗  Vehicle Analytics Dashboard")
    print(f"  Open: http://localhost:{settings.port}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start_server()
