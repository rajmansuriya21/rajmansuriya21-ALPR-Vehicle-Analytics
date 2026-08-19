"""
Video frame annotation.

Draws visual overlays on video frames including:
- Vehicle bounding boxes with track IDs
- Recognized plate text
- Entry/Exit virtual lines with labels
- Event notifications (ENTRY/EXIT with vehicle number)
- Status panel with real-time statistics
"""

import cv2
import numpy as np
from typing import Dict, List


# Colors (BGR)
COLOR_ENTRY = (0, 255, 0)       # Green
COLOR_EXIT = (0, 0, 255)        # Red
COLOR_VEHICLE_BOX = (255, 200, 0)  # Cyan-ish
COLOR_PLATE_TEXT = (0, 255, 255)   # Yellow
COLOR_PANEL_BG = (30, 30, 30)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SM = 0.5
FONT_SCALE_MD = 0.65
FONT_SCALE_LG = 0.8
FONT_THICKNESS = 2


class VideoAnnotator:
    """Draws visual annotations on video frames."""

    def __init__(self, line_detector=None):
        """
        Args:
            line_detector: LineCrossingDetector instance (for line coordinates).
        """
        self.line_detector = line_detector
        self._event_count = {"ENTRY": 0, "EXIT": 0}

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        resolved_plates: Dict[int, str],
        recent_events: List[Dict],
    ) -> np.ndarray:
        """
        Annotate a frame with all visual overlays.

        Args:
            frame: Original BGR frame.
            detections: List of vehicle detection dicts.
            resolved_plates: Map of track_id -> resolved plate text.
            recent_events: List of recent event dicts for on-screen display.

        Returns:
            Annotated frame (copy of original).
        """
        annotated = frame.copy()

        # 1. Draw virtual lines
        self._draw_lines(annotated)

        # 2. Draw vehicle detections
        for det in detections:
            self._draw_vehicle(annotated, det, resolved_plates)

        # 3. Draw event notifications
        self._draw_events(annotated, recent_events)

        # 4. Draw status panel
        self._draw_status_panel(annotated, detections, resolved_plates, recent_events)

        return annotated

    def _draw_lines(self, frame: np.ndarray):
        """Draw entry/exit virtual lines on the frame."""
        if self.line_detector is None:
            return

        lines = self.line_detector.get_lines_for_drawing()
        for line in lines:
            start = tuple(map(int, line["start"]))
            end = tuple(map(int, line["end"]))
            color = line["color"]
            label = line["label"]

            # Draw the line (thick, semi-transparent effect)
            cv2.line(frame, start, end, color, 3, cv2.LINE_AA)

            # Draw line label
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            label_pos = (mid_x - 30, mid_y - 10)

            # Background for label
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE_SM, 1)
            cv2.rectangle(
                frame,
                (label_pos[0] - 5, label_pos[1] - th - 5),
                (label_pos[0] + tw + 5, label_pos[1] + 5),
                color, -1,
            )
            cv2.putText(
                frame, label, label_pos,
                FONT, FONT_SCALE_SM, COLOR_WHITE, 1, cv2.LINE_AA,
            )

    def _draw_vehicle(
        self,
        frame: np.ndarray,
        det: Dict,
        resolved_plates: Dict[int, str],
    ):
        """Draw vehicle bounding box and plate text."""
        bbox = det["bbox"]
        track_id = det.get("track_id")
        x1, y1, x2, y2 = map(int, bbox)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VEHICLE_BOX, 2)

        # Track ID label
        if track_id is not None:
            label = f"ID:{track_id}"
            cv2.putText(
                frame, label, (x1, y1 - 5),
                FONT, FONT_SCALE_SM, COLOR_VEHICLE_BOX, 1, cv2.LINE_AA,
            )

        # Plate text (if resolved)
        if track_id and track_id in resolved_plates:
            plate_text = resolved_plates[track_id]
            # Draw plate text with background
            text_pos = (x1, y2 + 20)
            (tw, th), _ = cv2.getTextSize(plate_text, FONT, FONT_SCALE_MD, 2)
            cv2.rectangle(
                frame,
                (text_pos[0] - 3, text_pos[1] - th - 3),
                (text_pos[0] + tw + 3, text_pos[1] + 5),
                (0, 0, 0), -1,
            )
            cv2.putText(
                frame, plate_text, text_pos,
                FONT, FONT_SCALE_MD, COLOR_PLATE_TEXT, 2, cv2.LINE_AA,
            )

    def _draw_events(self, frame: np.ndarray, recent_events: List[Dict]):
        """Draw recent event notifications on the frame."""
        h, w = frame.shape[:2]

        for i, event in enumerate(recent_events[-3:]):  # Show last 3
            text = event.get("text", "")
            event_type = event.get("type", "")
            color = COLOR_ENTRY if event_type == "ENTRY" else COLOR_EXIT

            y_pos = 40 + i * 35

            # Background bar
            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE_MD, 2)
            cv2.rectangle(
                frame,
                (w - tw - 30, y_pos - th - 5),
                (w - 10, y_pos + 8),
                color, -1,
            )
            cv2.putText(
                frame, text, (w - tw - 20, y_pos),
                FONT, FONT_SCALE_MD, COLOR_WHITE, 2, cv2.LINE_AA,
            )

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        resolved_plates: Dict[int, str],
        recent_events: List[Dict],
    ):
        """Draw status panel in the bottom-left corner."""
        h, w = frame.shape[:2]

        # Count events
        vehicles_tracked = sum(1 for d in detections if d.get("track_id"))

        # Panel background
        panel_h = 80
        panel_w = 260
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (5, h - panel_h - 5),
            (panel_w + 5, h - 5),
            COLOR_PANEL_BG, -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Panel text
        y_base = h - panel_h + 15
        cv2.putText(
            frame, "Vehicle Analytics",
            (15, y_base),
            FONT, FONT_SCALE_SM, COLOR_WHITE, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"Tracking: {vehicles_tracked} vehicles",
            (15, y_base + 22),
            FONT, FONT_SCALE_SM, COLOR_GRAY, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"Plates: {len(resolved_plates)} recognized",
            (15, y_base + 44),
            FONT, FONT_SCALE_SM, COLOR_PLATE_TEXT, 1, cv2.LINE_AA,
        )
