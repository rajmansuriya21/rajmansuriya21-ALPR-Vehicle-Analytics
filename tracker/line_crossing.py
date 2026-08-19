"""
Virtual line crossing detection.

Supports two modes:
1. Two-line mode: Separate Entry and Exit lines.
2. Single-line + direction mode: One line with direction-based classification.

Detects when a tracked vehicle's centroid path crosses a virtual line
by testing segment intersection between consecutive centroid positions.
"""

import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict

from utils.geometry import segments_intersect, crossing_direction

logger = logging.getLogger(__name__)

# Minimum frames between events for the same track_id (prevents duplicates)
DEFAULT_COOLDOWN_FRAMES = 30


class LineCrossingDetector:
    """
    Detects when tracked objects cross virtual entry/exit lines.

    Maintains a history of centroid positions per track_id and checks
    for line segment intersection on each update.
    """

    def __init__(
        self,
        line_mode: str = "two_lines",
        entry_line: Tuple[int, int, int, int] = (550, 50, 550, 700),
        exit_line: Tuple[int, int, int, int] = (400, 50, 400, 700),
        single_line: Tuple[int, int, int, int] = (480, 50, 480, 700),
        entry_direction: str = "right_to_left",
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ):
        """
        Initialize line crossing detector.

        Args:
            line_mode: "two_lines" or "single_line_direction".
            entry_line: (x1, y1, x2, y2) for the entry line.
            exit_line: (x1, y1, x2, y2) for the exit line.
            single_line: (x1, y1, x2, y2) for single-line mode.
            entry_direction: Direction that counts as ENTRY in single-line mode.
            cooldown_frames: Minimum frames between events per track_id.
        """
        self.line_mode = line_mode
        self.cooldown_frames = cooldown_frames
        self.entry_direction = entry_direction

        # Parse line coordinates into point pairs.
        # _draw_* = original coordinates exactly as drawn (used for on-screen display)
        # _start/_end = extended to full frame span (used for crossing detection)
        if line_mode == "two_lines":
            # Save original drawn coords for display
            self._entry_draw_start = (entry_line[0], entry_line[1])
            self._entry_draw_end   = (entry_line[2], entry_line[3])
            self._exit_draw_start  = (exit_line[0],  exit_line[1])
            self._exit_draw_end    = (exit_line[2],  exit_line[3])
            # Extend for detection
            self.entry_line_start, self.entry_line_end = self._extend_line(*entry_line)
            self.exit_line_start,  self.exit_line_end  = self._extend_line(*exit_line)
            logger.info(
                f"Two-line mode: Entry={self.entry_line_start}->{self.entry_line_end}, "
                f"Exit={self.exit_line_start}->{self.exit_line_end}"
            )
        else:
            # Save original
            self._single_draw_start = (single_line[0], single_line[1])
            self._single_draw_end   = (single_line[2], single_line[3])
            # Extend for detection
            self.single_line_start, self.single_line_end = self._extend_line(*single_line)
            logger.info(
                f"Single-line mode: Line={self.single_line_start}->{self.single_line_end}, "
                f"Entry direction={entry_direction}"
            )

        # State: previous centroid per track_id
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}
        # State: last event frame per track_id (for cooldown)
        self._last_event_frame: Dict[int, int] = defaultdict(lambda: -999)

    def update(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        frame_idx: int,
    ) -> Optional[str]:
        """
        Update a tracked object's position and check for line crossing.

        Args:
            track_id: Persistent track ID from the tracker.
            centroid: Current (x, y) centroid of the vehicle.
            frame_idx: Current frame index (for cooldown).

        Returns:
            "ENTRY", "EXIT", or None if no crossing detected.
        """
        prev = self._prev_centroids.get(track_id)
        self._prev_centroids[track_id] = centroid

        # Need at least two positions to detect crossing
        if prev is None:
            return None

        # Check cooldown
        frames_since_last = frame_idx - self._last_event_frame[track_id]
        if frames_since_last < self.cooldown_frames:
            return None

        # Check for crossing based on mode
        if self.line_mode == "two_lines":
            event = self._check_two_lines(prev, centroid)
        else:
            event = self._check_single_line(prev, centroid)

        if event:
            self._last_event_frame[track_id] = frame_idx
            logger.debug(f"Track {track_id}: {event} detected at frame {frame_idx}")

        return event

    def _check_two_lines(
        self,
        prev: Tuple[float, float],
        curr: Tuple[float, float],
    ) -> Optional[str]:
        """Check crossing against separate entry and exit lines."""
        # Check entry line
        if segments_intersect(prev, curr, self.entry_line_start, self.entry_line_end):
            return "ENTRY"

        # Check exit line
        if segments_intersect(prev, curr, self.exit_line_start, self.exit_line_end):
            return "EXIT"

        return None

    def _check_single_line(
        self,
        prev: Tuple[float, float],
        curr: Tuple[float, float],
    ) -> Optional[str]:
        """Check crossing against a single line with direction detection."""
        if not segments_intersect(prev, curr, self.single_line_start, self.single_line_end):
            return None

        # Determine crossing direction
        direction = crossing_direction(
            prev, curr, self.single_line_start, self.single_line_end
        )

        if direction == self.entry_direction:
            return "ENTRY"
        else:
            return "EXIT"

    def get_lines_for_drawing(self) -> list:
        """
        Return line definitions for the video annotator.
        Uses ORIGINAL drawn coordinates so the line appears exactly where the user drew it.

        Returns:
            List of dicts with keys: start, end, color, label.
        """
        lines = []
        if self.line_mode == "two_lines":
            lines.append({
                "start": self._entry_draw_start,
                "end":   self._entry_draw_end,
                "color": (0, 255, 0),   # Green for entry
                "label": "ENTRY",
            })
            lines.append({
                "start": self._exit_draw_start,
                "end":   self._exit_draw_end,
                "color": (0, 0, 255),   # Red for exit
                "label": "EXIT",
            })
        else:
            lines.append({
                "start": self._single_draw_start,
                "end":   self._single_draw_end,
                "color": (255, 255, 0),  # Cyan for single line
                "label": f"LINE ({self.entry_direction}→ENTRY)",
            })
        return lines

    def update_lines(
        self,
        entry_line: Optional[Tuple[int, int, int, int]] = None,
        exit_line: Optional[Tuple[int, int, int, int]] = None,
        single_line: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Update line coordinates (used by interactive line drawer or API)."""
        if entry_line and self.line_mode == "two_lines":
            self._entry_draw_start = (entry_line[0], entry_line[1])
            self._entry_draw_end   = (entry_line[2], entry_line[3])
            self.entry_line_start, self.entry_line_end = self._extend_line(*entry_line)
        if exit_line and self.line_mode == "two_lines":
            self._exit_draw_start = (exit_line[0], exit_line[1])
            self._exit_draw_end   = (exit_line[2], exit_line[3])
            self.exit_line_start, self.exit_line_end = self._extend_line(*exit_line)
        if single_line and self.line_mode == "single_line_direction":
            self._single_draw_start = (single_line[0], single_line[1])
            self._single_draw_end   = (single_line[2], single_line[3])
            self.single_line_start, self.single_line_end = self._extend_line(*single_line)

    def reset(self):
        """Reset tracking state (for processing a new video)."""
        self._prev_centroids.clear()
        self._last_event_frame.clear()

    @staticmethod
    def _extend_line(
        x1: int, y1: int, x2: int, y2: int,
        frame_w: int = 1280, frame_h: int = 720,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Extend a drawn line segment to span the full frame.

        For a mostly-vertical line (|dx| < |dy|), extend to frame height (y: 0–frame_h).
        For a mostly-horizontal line (|dx| >= |dy|), extend to frame width (x: 0–frame_w).

        This ensures that lines drawn short by the user (e.g. only halfway down the
        frame) still trigger crossings anywhere along their axis.
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dy == 0 and dx == 0:
            # Degenerate point — return as-is
            return (x1, y1), (x2, y2)

        if dy >= dx:
            # Mostly vertical — extend to full height
            # x at y=0: x = x1 + (0 - y1) * (x2-x1)/(y2-y1)
            if y2 != y1:
                slope_x = (x2 - x1) / (y2 - y1)
                new_x1 = int(x1 + (0 - y1) * slope_x)
                new_x2 = int(x1 + (frame_h - y1) * slope_x)
            else:
                new_x1, new_x2 = x1, x2
            return (new_x1, 0), (new_x2, frame_h)
        else:
            # Mostly horizontal — extend to full width
            if x2 != x1:
                slope_y = (y2 - y1) / (x2 - x1)
                new_y1 = int(y1 + (0 - x1) * slope_y)
                new_y2 = int(y1 + (frame_w - x1) * slope_y)
            else:
                new_y1, new_y2 = y1, y2
            return (0, new_y1), (frame_w, new_y2)
