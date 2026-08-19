# Vehicle Entry & Exit Analytics System (ALPR)

An AI-powered application that analyzes CCTV footage and automatically creates entry/exit logs of vehicle license plate numbers, with configurable virtual detection lines, visit history tracking, and LLM-powered analytics.

## Architecture

```
ALPR/
├── main.py                 # CLI entry point (python main.py)
├── .env                    # All configuration (no code changes needed)
├── config/settings.py      # Pydantic-settings configuration loader
├── models/
│   ├── detector.py         # YOLO11 vehicle + plate detection
│   └── ocr.py              # PaddleOCR license plate recognition
├── tracker/
│   ├── vehicle_tracker.py  # BoTSORT multi-object tracker
│   └── line_crossing.py    # Virtual line crossing detection
├── core/
│   ├── pipeline.py         # Main video processing pipeline
│   ├── event_manager.py    # Entry/Exit event creation
│   ├── plate_aggregator.py # Majority-vote OCR consensus
│   └── visit_store.py      # Visit lifecycle management
├── output/
│   ├── video_annotator.py  # Visual annotations on frames
│   ├── json_logger.py      # JSON event log
│   ├── csv_reporter.py     # CSV visit report
│   └── ai_analytics.py     # Gemini LLM analytics report
├── utils/
│   ├── line_drawer.py      # Interactive mouse line drawing
│   ├── geometry.py         # Line intersection math
│   └── video_io.py         # Video reader/writer
├── app/
│   ├── server.py           # FastAPI web dashboard
│   └── static/             # Dashboard frontend
└── test/                   # Sample videos for evaluation
```

## Features

- **Single-Stage Plate Detection**: Highly optimized pipeline using YOLO11 to detect and track license plates directly, bypassing the slower vehicle-first two-stage approach.
- **License Plate OCR**: PaddleOCR with CLAHE contrast enhancement for accurate text reading.
- **Configurable Detection Lines**: Two-line mode (separate entry/exit) or single-line with direction
- **Interactive Line Drawing**: Draw lines with mouse on first frame, or set coordinates in .env
- **Visit History Management**: Entry → Exit → Re-entry lifecycle with duration tracking
- **Plate Text Consensus**: Weighted majority voting across frames for reliable OCR
- **Annotated Output Video**: Bounding boxes, plate text, entry/exit lines, event overlays
- **JSON Event Log**: Structured events with vehicle number, event type, timestamp, camera
- **CSV Visit Report**: Vehicle Number, Entry Time, Exit Time, Duration, Visit No., Status
- **AI Analytics Report**: Gemini LLM-powered analysis of event patterns
- **Web Dashboard**: FastAPI backend with real-time processing view and analytics charts

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and update:
- `VIDEO_SOURCE`: Path to your CCTV video
- `ENTRY_LINE` / `EXIT_LINE`: Virtual line coordinates (x1,y1,x2,y2)
- `LLM_API_KEY`: Google Gemini API key (for AI analytics report)

### 3. Run (CLI Mode)

```bash
python main.py
```

### 4. Run (Web Dashboard)

```bash
python -m app.server
# Open http://localhost:8000
```

## Configuration Reference

All settings are in `.env`. No code changes required.

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_SOURCE` | Video file or directory path | `test/1000267609.mp4` |
| `CAMERA_ID` | Camera identifier for logs | `camera_1` |
| `VEHICLE_MODEL` | YOLO11 model (auto-downloads) | `yolo11n.pt` |
| `PLATE_MODEL` | Plate YOLO model (empty = PaddleOCR) | `` |
| `LINE_MODE` | `two_lines` or `single_line_direction` | `two_lines` |
| `INTERACTIVE_LINE_DRAW` | `true` to draw lines with mouse | `false` |
| `ENTRY_LINE` | Entry line coordinates (x1,y1,x2,y2) | `550,50,550,700` |
| `EXIT_LINE` | Exit line coordinates (x1,y1,x2,y2) | `400,50,400,700` |
| `ENTRY_DIRECTION` | Direction for single-line mode | `right_to_left` |
| `LLM_PROVIDER` | `gemini` or `openai` | `gemini` |
| `LLM_API_KEY` | API key for analytics report | |
| `FRAME_SKIP` | Process every Nth frame | `2` |
| `DEVICE` | `cpu` or `cuda:0` | `cpu` |

## Detection Pipeline

```
Frame → YOLO11 Plate Detection (license_plate_detector.pt) → BoTSORT Tracking →
  Per Plate:
    → PaddleOCR Text Reading → Majority Vote Aggregation →
    → Line Crossing Check → Event Generation →
    → Visit Store Update
  → Frame Annotation → Output Video
```
*Note: The pipeline is built to support both single-stage (plate-only) and two-stage (vehicle → plate) detection, but defaults to the highly optimized single-stage approach for maximum FPS.*

## Entry/Exit Logic

- **Two-line mode**: Separate entry and exit lines. Vehicle crossing entry line → ENTRY event; crossing exit line → EXIT event.
- **Single-line mode**: One line with direction detection using cross-product. Configurable which direction is ENTRY.
- **Visit lifecycle**: ENTRY creates visit (status: Inside) → EXIT updates with duration (status: Completed) → Re-ENTRY creates new visit with incremented visit_no.

## Output Files

| File | Description |
|------|-------------|
| `output_data/annotated_*.mp4` | Video with bboxes, plates, lines, events |
| `output_data/event_log_*.json` | JSON array of all detected events |
| `output_data/visit_report_*.csv` | CSV visit report with durations |
| `output_data/ai_analytics_*.md` | LLM-generated analytics report |

## Technology Stack

- **Detection & Tracking**: YOLO11 (Ultralytics) + BoTSORT

- **Plate OCR**: PaddleOCR with CLAHE preprocessing
- **Configuration**: pydantic-settings + python-dotenv
- **Web Dashboard**: FastAPI + vanilla HTML/CSS/JS + Chart.js
- **AI Analytics**: Google Gemini API
- **Language**: Python 3.10+

## License

MIT
