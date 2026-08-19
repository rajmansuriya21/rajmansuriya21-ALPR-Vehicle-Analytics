"""
Main video processing pipeline.

Orchestrates the complete frame-by-frame processing flow:
Frame → Vehicle Detection → Tracking → Plate Detection/OCR →
Plate Aggregation → Line Crossing → Event Generation → Annotation → Output

Supports callbacks for real-time streaming (WebSocket integration).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict

import cv2
import numpy as np

from config.settings import Settings
from models.detector import VehicleDetector, PlateDetector
from models.ocr import PlateOCR
from tracker.vehicle_tracker import VehicleTracker
from tracker.line_crossing import LineCrossingDetector
from core.plate_aggregator import PlateAggregator
from core.event_manager import EventManager, VehicleEvent
from core.visit_store import VisitStore
from output.video_annotator import VideoAnnotator
from output.json_logger import JsonLogger
from output.csv_reporter import CsvReporter
from output.ai_analytics import AiAnalytics
from utils.video_io import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    End-to-end video processing pipeline for vehicle analytics.

    Processes video frames sequentially, detecting vehicles, reading
    license plates, tracking movement, and generating entry/exit events
    when vehicles cross configured virtual lines.
    """

    def __init__(
        self,
        settings: Settings,
        on_event: Optional[Callable[[dict], None]] = None,
        on_frame: Optional[Callable[[np.ndarray, int, int], None]] = None,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
    ):
        """
        Initialize the pipeline with all components.

        Args:
            settings: Application configuration.
            on_event: Callback for each new event: fn(event_dict).
            on_frame: Callback for annotated frames: fn(frame, frame_idx, total).
            on_progress: Callback for progress: fn(current, total, percent).
        """
        self.settings = settings
        self.on_event = on_event
        self.on_frame = on_frame
        self.on_progress = on_progress
        self._is_running = False
        self._should_stop = False

        # Initialize components
        logger.info("Initializing pipeline components...")

        # Detection models
        self.vehicle_detector = VehicleDetector(
            model_path=settings.vehicle_model,
            device=settings.device,
            confidence=settings.confidence_threshold,
            tracker_type=settings.tracker_type,
        )
        self.plate_detector = PlateDetector(
            model_path=settings.plate_model,
            device=settings.device,
            confidence=settings.plate_confidence_threshold,
        )
        self.ocr = PlateOCR(
            lang=settings.ocr_lang,
            confidence_threshold=settings.ocr_confidence_threshold,
        )

        # Tracker
        self.vehicle_tracker = VehicleTracker()

        # Core logic
        self.plate_aggregator = PlateAggregator()
        self.visit_store = VisitStore()
        self.event_manager = EventManager(
            visit_store=self.visit_store,
            camera_id=settings.camera_id,
            on_event=self._handle_event,
        )

        # Output (line_detector set in _init_line_detector below)
        self.video_annotator = VideoAnnotator(line_detector=None)
        self.json_logger = None  # Initialized per video
        self.csv_reporter = None

        # Per-track pending crossings: track_id -> {crossing, timestamp, frame_idx}
        self._pending_crossings: Dict[int, dict] = {}

        # Line crossing detector — initialized LAST so video_annotator exists.
        # Detection coordinates come from .env settings (horizontal defaults).
        # Visual display coordinates can be overridden via set_display_lines().
        self._init_line_detector()

        logger.info("Pipeline initialized successfully.")

    def _init_line_detector(self):
        """Initialize the line crossing detector based on settings."""
        s = self.settings
        if s.line_mode == "two_lines":
            self.line_detector = LineCrossingDetector(
                line_mode="two_lines",
                entry_line=s.get_entry_line_coords(),
                exit_line=s.get_exit_line_coords(),
            )
        else:
            self.line_detector = LineCrossingDetector(
                line_mode="single_line_direction",
                single_line=s.get_single_line_coords(),
                entry_direction=s.entry_direction,
            )
        # Update the annotator to use the new detector
        if self.video_annotator is not None:
            self.video_annotator.line_detector = self.line_detector
        logger.info(
            f"Line detector: entry={s.entry_line}, exit={s.exit_line}, mode={s.line_mode}"
        )

    def set_display_lines(
        self,
        entry_line=None,
        exit_line=None,
        single_line=None,
    ):
        """
        Override the VISUAL display coordinates of the virtual lines
        without changing the DETECTION coordinates.

        Use this when the user drew custom lines (e.g. vertical) that should
        be shown on-screen, while detection still uses the internally configured
        coordinates (e.g. horizontal lines tuned for this camera angle).

        Args:
            entry_line: (x1, y1, x2, y2) tuple or None.
            exit_line:  (x1, y1, x2, y2) tuple or None.
            single_line: (x1, y1, x2, y2) tuple or None.
        """
        if self.line_detector is None:
            return
        ld = self.line_detector
        if entry_line and hasattr(ld, '_entry_draw_start'):
            ld._entry_draw_start = (entry_line[0], entry_line[1])
            ld._entry_draw_end   = (entry_line[2], entry_line[3])
        if exit_line and hasattr(ld, '_exit_draw_start'):
            ld._exit_draw_start = (exit_line[0], exit_line[1])
            ld._exit_draw_end   = (exit_line[2], exit_line[3])
        if single_line and hasattr(ld, '_single_draw_start'):
            ld._single_draw_start = (single_line[0], single_line[1])
            ld._single_draw_end   = (single_line[2], single_line[3])

    def process_video(self, video_path: str) -> Dict:
        """
        Process a single video file through the complete pipeline.

        Args:
            video_path: Path to the input video.

        Returns:
            Results dict with events, visits, summary, and output paths.
        """
        self._is_running = True
        self._should_stop = False

        logger.info(f"Processing video: {video_path}")

        # Setup
        reader = VideoReader(video_path)
        output_dir = self.settings.ensure_output_dir()
        video_name = Path(video_path).stem

        # Initialize output writers
        output_video_path = str(output_dir / f"annotated_{video_name}.mp4")
        writer = None
        if self.settings.save_annotated_video:
            output_fps = reader.fps / max(1, self.settings.frame_skip)
            writer = VideoWriter(
                output_video_path, output_fps, reader.width, reader.height
            )

        json_log_path = str(output_dir / f"event_log_{video_name}.json")
        if self.settings.save_json_log:
            self.json_logger = JsonLogger(json_log_path)

        # Processing start time (for timestamp calculation)
        start_time = datetime.now()

        # Track plate text resolved per track_id for annotation
        resolved_plates: Dict[int, str] = {}
        # Recent events for on-screen display
        recent_events = []

        logger.info(
            f"Video: {reader.width}x{reader.height} @ {reader.fps:.1f} FPS, "
            f"{reader.total_frames} frames"
        )

        # ── Frame-by-frame processing ─────────────────────
        for frame_idx, frame in reader:
            if self._should_stop:
                logger.info("Processing stopped by user.")
                break

            # Skip frames for performance
            if frame_idx % self.settings.frame_skip != 0:
                continue

            # 1. Detect and track vehicles
            detections = self.vehicle_detector.detect_and_track(frame)
            detections = self.vehicle_tracker.update(detections, frame_idx)

            # 2. Process each tracked object
            for det in detections:
                track_id = det.get("track_id")
                if track_id is None:
                    continue

                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)

                # Clamp to frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                crop = frame[y1:y2, x1:x2]

                if self.plate_detector.enabled:
                    # Two-stage mode: primary is vehicle, so find plate inside
                    plate_dets = self.plate_detector.detect_in_vehicle(frame, bbox)
                    for pd in plate_dets:
                        px1, py1, px2, py2 = map(int, pd["bbox"])
                        px1, py1 = max(0, px1), max(0, py1)
                        px2, py2 = min(w, px2), min(h, py2)
                        if px2 - px1 < 5 or py2 - py1 < 5:
                            continue
                        plate_crop = frame[py1:py2, px1:px2]
                        text, conf = self.ocr.read_plate(plate_crop)
                        if text:
                            self.plate_aggregator.add_reading(track_id, text, conf)
                else:
                    # Single-stage mode or fallback
                    if det.get("class_name") == "license_plate" or "plate" in self.settings.vehicle_model.lower():
                        # The tracked object IS the plate
                        text, conf = self.ocr.read_plate(crop)
                        if text:
                            self.plate_aggregator.add_reading(track_id, text, conf)
                    else:
                        # Fallback: primary is vehicle but no dedicated plate model, let PaddleOCR find the plate
                        plates = self.ocr.detect_and_read(crop)
                        for plate in plates:
                            self.plate_aggregator.add_reading(
                                track_id, plate["text"], plate["confidence"]
                            )

                # Update resolved plate cache for annotation
                resolved = self.plate_aggregator.get_plate(track_id)
                if resolved:
                    resolved_plates[track_id] = resolved
                    
                    # If this track_id has a pending crossing, fire it now!
                    if track_id in self._pending_crossings:
                        pending = self._pending_crossings.pop(track_id)
                        logger.info(
                            f"Delayed event: {pending['crossing']} for {resolved} "
                            f"(track {track_id}, queued at frame {pending['frame_idx']})"
                        )
                        event = self.event_manager.create_event(
                            vehicle_number=resolved,
                            event_type=pending["crossing"],
                            timestamp=pending["timestamp"],
                            track_id=track_id,
                        )
                        if self.json_logger:
                            self.json_logger.log_event(event.to_dict())
                        time_only = pending["timestamp"].split("T")[1] if "T" in pending["timestamp"] else pending["timestamp"]
                        recent_events.append({
                            "text": f"[{time_only}] {pending['crossing']}: {resolved}",
                            "time": pending["timestamp"],
                            "type": pending["crossing"],
                            "frame": pending["frame_idx"],
                        })
                        if len(recent_events) > 5:
                            recent_events.pop(0)

                # 3. Check line crossing — use BOTTOM edge of plate for reliable head-on detection
                # Bottom-center of the bbox is (center_x, y2) which hits a horizontal line first
                bottom_center = ((x1 + x2) / 2.0, float(y2))
                if bottom_center:
                    crossing = self.line_detector.update(
                        track_id, bottom_center, frame_idx
                    )

                    if crossing:
                        plate_number = resolved_plates.get(track_id)
                        timestamp = EventManager.frame_to_timestamp(
                            frame_idx, reader.fps, start_time
                        )
                        logger.info(
                            f"Line crossed: track={track_id}, event={crossing}, "
                            f"plate={'KNOWN:'+plate_number if plate_number else 'PENDING'}, "
                            f"frame={frame_idx}, bottom_y={y2}"
                        )
                        
                        if plate_number:
                            # Plate already resolved — log event immediately
                            event = self.event_manager.create_event(
                                vehicle_number=plate_number,
                                event_type=crossing,
                                timestamp=timestamp,
                                track_id=track_id,
                            )
                            if self.json_logger:
                                self.json_logger.log_event(event.to_dict())
                            time_only = timestamp.split("T")[1] if "T" in timestamp else timestamp
                            recent_events.append({
                                "text": f"[{time_only}] {crossing}: {plate_number}",
                                "time": timestamp,
                                "type": crossing,
                                "frame": frame_idx,
                            })
                            if len(recent_events) > 5:
                                recent_events.pop(0)
                        else:
                            # Plate not yet resolved — store crossing by track_id
                            # It will be fired as soon as OCR resolves this track
                            self._pending_crossings[track_id] = {
                                "crossing": crossing,
                                "timestamp": timestamp,
                                "frame_idx": frame_idx,
                            }

            # 4. Annotate frame
            annotated = self.video_annotator.annotate(
                frame, detections, resolved_plates, recent_events
            )

            # 5. Write to output video and show live display
            if writer:
                writer.write(annotated)
                
            # Show live inference
            cv2.imshow("Live Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Live display stopped by user.")
                self.stop()

            # 6. Callbacks
            if self.on_frame and frame_idx % (self.settings.frame_skip * 5) == 0:
                self.on_frame(annotated, frame_idx, reader.total_frames)

            if self.on_progress and frame_idx % 10 == 0:
                percent = (frame_idx / reader.total_frames) * 100
                self.on_progress(frame_idx, reader.total_frames, percent)

            # Clean up old recent events (older than 90 frames)
            recent_events = [
                e for e in recent_events
                if frame_idx - e.get("frame", 0) < 90
            ]

        # ── Finalize ──────────────────────────────────────
        reader.release()
        cv2.destroyAllWindows()
        if writer:
            writer.release()
            logger.info(f"Annotated video saved: {output_video_path}")

        if self.json_logger:
            self.json_logger.save()
            logger.info(f"JSON event log saved: {json_log_path}")

        # Generate CSV report
        csv_path = str(output_dir / f"visit_report_{video_name}.csv")
        if self.settings.save_csv_report:
            self.csv_reporter = CsvReporter(csv_path)
            self.csv_reporter.generate(self.visit_store)
            logger.info(f"CSV report saved: {csv_path}")

        # Generate AI analytics report
        ai_report_path = str(output_dir / f"ai_analytics_{video_name}.md")
        if self.settings.generate_ai_report and self.settings.llm_api_key:
            try:
                ai = AiAnalytics(
                    provider=self.settings.llm_provider,
                    api_key=self.settings.llm_api_key,
                    model=self.settings.llm_model,
                )
                events = self.event_manager.get_all_events()
                summary = self.visit_store.get_summary()
                ai.generate_report(events, summary, ai_report_path)
                logger.info(f"AI analytics report saved: {ai_report_path}")
            except Exception as e:
                logger.error(f"AI analytics generation failed: {e}")

        self._is_running = False

        # Compile results
        results = {
            "video_path": video_path,
            "events": self.event_manager.get_all_events(),
            "visits": [v.to_dict() for v in self.visit_store.get_all_visits()],
            "summary": self.visit_store.get_summary(),
            "output_paths": {
                "annotated_video": output_video_path if self.settings.save_annotated_video else None,
                "json_log": json_log_path if self.settings.save_json_log else None,
                "csv_report": csv_path if self.settings.save_csv_report else None,
                "ai_report": ai_report_path if self.settings.generate_ai_report else None,
            },
        }

        logger.info(
            f"Processing complete. Events: {len(results['events'])}, "
            f"Visits: {len(results['visits'])}, "
            f"Summary: {results['summary']}"
        )

        return results

    def _handle_event(self, event: VehicleEvent):
        """Internal event handler — forwards to external callback."""
        if self.on_event:
            self.on_event(event.to_dict())

    def stop(self):
        """Signal the pipeline to stop processing."""
        self._should_stop = True

    @property
    def is_running(self) -> bool:
        return self._is_running

    def update_lines(self, **kwargs):
        """Update line coordinates (from interactive drawer or API)."""
        self.line_detector.update_lines(**kwargs)

    def reset(self):
        """Reset all pipeline state for a new processing run."""
        self.vehicle_tracker.reset()
        if self.line_detector:
            self.line_detector.reset()
        self.plate_aggregator.reset()
        self.visit_store.reset()
        self.event_manager.reset()
        self._pending_crossings.clear()
