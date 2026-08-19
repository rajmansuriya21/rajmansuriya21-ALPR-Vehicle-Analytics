"""
Interactive line drawing tool.

Opens the first frame of the video in a GUI window and allows the user
to draw entry/exit lines by clicking points with the mouse.

Controls:
- Left click: Add a point (2 clicks = 1 line)
- 'r': Reset current line
- 'n': Move to next line (Entry → Exit)
- 'q' or Enter: Confirm and close
- ESC: Cancel
"""

import logging
from typing import Tuple, Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class InteractiveLineDrawer:
    """
    GUI tool for drawing virtual entry/exit lines on a video frame.

    Opens an OpenCV window showing the first frame of the video.
    The user clicks to define line endpoints.
    """

    def __init__(self, frame: np.ndarray, line_mode: str = "two_lines"):
        """
        Args:
            frame: First frame of the video (BGR).
            line_mode: "two_lines" or "single_line_direction".
        """
        self.frame = frame.copy()
        self.display = frame.copy()
        self.line_mode = line_mode
        self.points: List[Tuple[int, int]] = []
        self.lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.current_line_idx = 0
        self.done = False

        if line_mode == "two_lines":
            self.line_labels = ["ENTRY Line (Green)", "EXIT Line (Red)"]
            self.line_colors = [(0, 255, 0), (0, 0, 255)]
            self.total_lines = 2
        else:
            self.line_labels = ["Detection Line (Cyan)"]
            self.line_colors = [(255, 255, 0)]
            self.total_lines = 1

    def draw(self) -> Optional[dict]:
        """
        Open the interactive drawing window.

        Returns:
            Dict with line coordinates, or None if cancelled.
            For two_lines: {"entry_line": (x1,y1,x2,y2), "exit_line": (x1,y1,x2,y2)}
            For single_line: {"single_line": (x1,y1,x2,y2)}
        """
        window_name = "Draw Lines - Click to add points"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while not self.done:
            self._render()
            cv2.imshow(window_name, self.display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 13:  # 'q' or Enter
                if len(self.lines) >= self.total_lines:
                    self.done = True
                else:
                    logger.warning(
                        f"Need {self.total_lines} lines, have {len(self.lines)}. "
                        "Keep drawing."
                    )
            elif key == ord("r"):  # Reset current
                self.points.clear()
            elif key == ord("n"):  # Next line
                if len(self.points) == 2:
                    self.lines.append((self.points[0], self.points[1]))
                    self.points.clear()
                    self.current_line_idx += 1
            elif key == 27:  # ESC - cancel
                cv2.destroyWindow(window_name)
                return None

        cv2.destroyWindow(window_name)

        # Build result
        if self.line_mode == "two_lines" and len(self.lines) >= 2:
            entry = self.lines[0]
            exit_ = self.lines[1]
            return {
                "entry_line": (entry[0][0], entry[0][1], entry[1][0], entry[1][1]),
                "exit_line": (exit_[0][0], exit_[0][1], exit_[1][0], exit_[1][1]),
            }
        elif self.line_mode == "single_line_direction" and len(self.lines) >= 1:
            line = self.lines[0]
            return {
                "single_line": (line[0][0], line[0][1], line[1][0], line[1][1]),
            }

        return None

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

            if len(self.points) == 2:
                self.lines.append((self.points[0], self.points[1]))
                self.points.clear()
                self.current_line_idx += 1

                if self.current_line_idx >= self.total_lines:
                    self.done = True

    def _render(self):
        """Render the current state on the display frame."""
        self.display = self.frame.copy()

        # Draw completed lines
        for i, (p1, p2) in enumerate(self.lines):
            color = self.line_colors[i % len(self.line_colors)]
            cv2.line(self.display, p1, p2, color, 3, cv2.LINE_AA)
            label = self.line_labels[i] if i < len(self.line_labels) else f"Line {i+1}"
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                self.display, label, (mid[0] - 50, mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

        # Draw current points
        for pt in self.points:
            cv2.circle(self.display, pt, 6, (0, 255, 255), -1)

        # Instructions overlay
        h, w = self.display.shape[:2]
        instructions = [
            "Left Click: Add point (2 points = 1 line)",
            "'R': Reset current line",
            "'Q' or Enter: Confirm | ESC: Cancel",
        ]
        if self.current_line_idx < self.total_lines:
            label = self.line_labels[self.current_line_idx]
            instructions.insert(0, f"Drawing: {label}")

        for i, text in enumerate(instructions):
            y = h - 20 - (len(instructions) - 1 - i) * 25
            cv2.putText(
                self.display, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
