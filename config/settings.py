"""
Application configuration loaded from .env file.

All configurable values are defined here using pydantic-settings.
The application reads these values at startup — no code changes required.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Tuple
from pathlib import Path


class Settings(BaseSettings):
    """Central configuration class. All values loaded from .env file."""

    # ── Input ──────────────────────────────────────────────
    video_source: str = Field(
        default="test/sample_gate_video.mp4",
        description="Path to input video file or directory containing videos",
    )
    camera_id: str = Field(
        default="camera_1",
        description="Camera identifier used in event logs",
    )

    # ── Models ─────────────────────────────────────────────
    vehicle_model: str = Field(
        default="license_plate_detector.pt",
        description="Primary YOLO model for detection (e.g. license_plate_detector.pt)",
    )
    plate_model: str = Field(
        default="",
        description="Secondary YOLO model for license plate detection (empty = single stage)",
    )
    ocr_lang: str = Field(default="en", description="PaddleOCR language")
    device: str = Field(default="cpu", description="Inference device: cpu or cuda:0")

    # ── Line Configuration ─────────────────────────────────
    line_mode: str = Field(
        default="two_lines",
        description="Detection mode: 'two_lines' or 'single_line_direction'",
    )
    interactive_line_draw: bool = Field(
        default=False,
        description="If true, opens GUI to draw lines interactively with mouse",
    )
    entry_line: str = Field(
        default="550,50,550,700",
        description="Entry line coordinates as x1,y1,x2,y2",
    )
    exit_line: str = Field(
        default="400,50,400,700",
        description="Exit line coordinates as x1,y1,x2,y2",
    )
    single_line: str = Field(
        default="480,50,480,700",
        description="Single line coordinates as x1,y1,x2,y2 (for single_line_direction mode)",
    )
    entry_direction: str = Field(
        default="right_to_left",
        description="Direction that counts as ENTRY: 'left_to_right' or 'right_to_left'",
    )

    # ── Output ─────────────────────────────────────────────
    output_dir: str = Field(default="output_data", description="Directory for outputs")
    save_annotated_video: bool = Field(default=True)
    save_json_log: bool = Field(default=True)
    save_csv_report: bool = Field(default=True)
    generate_ai_report: bool = Field(default=True)

    # ── LLM ────────────────────────────────────────────────
    llm_provider: str = Field(default="gemini", description="LLM provider: gemini or openai")
    llm_api_key: str = Field(default="", description="API key for LLM provider")
    llm_model: str = Field(default="gemini-2.0-flash", description="LLM model name")

    # ── Processing ─────────────────────────────────────────
    confidence_threshold: float = Field(default=0.4, description="Vehicle detection confidence")
    plate_confidence_threshold: float = Field(default=0.3, description="Plate detection confidence")
    ocr_confidence_threshold: float = Field(default=0.5, description="OCR minimum confidence")
    tracker_type: str = Field(default="botsort", description="Tracker: botsort or bytetrack")
    frame_skip: int = Field(default=2, description="Process every Nth frame")

    # ── Web Server ─────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Computed Properties ────────────────────────────────

    def get_entry_line_coords(self) -> Tuple[int, int, int, int]:
        """Parse entry line string into (x1, y1, x2, y2) tuple."""
        parts = [int(x.strip()) for x in self.entry_line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"ENTRY_LINE must have 4 values, got {len(parts)}: {self.entry_line}")
        return tuple(parts)

    def get_exit_line_coords(self) -> Tuple[int, int, int, int]:
        """Parse exit line string into (x1, y1, x2, y2) tuple."""
        parts = [int(x.strip()) for x in self.exit_line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"EXIT_LINE must have 4 values, got {len(parts)}: {self.exit_line}")
        return tuple(parts)

    def get_single_line_coords(self) -> Tuple[int, int, int, int]:
        """Parse single line string into (x1, y1, x2, y2) tuple."""
        parts = [int(x.strip()) for x in self.single_line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"SINGLE_LINE must have 4 values, got {len(parts)}: {self.single_line}")
        return tuple(parts)

    def get_video_paths(self) -> list:
        """Return list of video file paths. Supports single file or directory."""
        source = Path(self.video_source)
        if source.is_file():
            return [str(source)]
        elif source.is_dir():
            extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
            videos = [str(f) for f in source.iterdir() if f.suffix.lower() in extensions]
            if not videos:
                raise FileNotFoundError(f"No video files found in directory: {source}")
            return sorted(videos)
        else:
            raise FileNotFoundError(f"Video source not found: {source}")

    def ensure_output_dir(self) -> Path:
        """Create output directory if it doesn't exist and return path."""
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out


def load_settings() -> Settings:
    """Load and return application settings from .env file."""
    return Settings()
