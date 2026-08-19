"""
Video I/O utilities.

Provides wrapper classes for OpenCV video reading and writing
with metadata extraction and codec management.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Iterator, Tuple


class VideoReader:
    """
    Iterator-based video reader with metadata.

    Usage:
        reader = VideoReader("input.mp4")
        for frame_idx, frame in reader:
            process(frame)
        reader.release()
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_idx = 0

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, frame) tuples."""
        self._frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            idx = self._frame_idx
            self._frame_idx += 1
            yield idx, frame

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame. Returns None at end of video."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        self._frame_idx += 1
        return frame

    def get_first_frame(self) -> Optional[np.ndarray]:
        """Read the first frame without advancing the iterator."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame if ret else None

    def release(self):
        """Release the video capture resource."""
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoReader(path='{self.path}', "
            f"fps={self.fps:.1f}, frames={self.total_frames}, "
            f"resolution={self.width}x{self.height})"
        )


class VideoWriter:
    """
    Video writer wrapper with automatic codec selection.

    Usage:
        writer = VideoWriter("output.mp4", fps=30.0, width=1920, height=1080)
        writer.write(frame)
        writer.release()
    """

    # Codec priority: try H.264 first, fall back to mp4v
    _CODEC_OPTIONS = [
        ("mp4v", ".mp4"),
        ("XVID", ".avi"),
    ]

    def __init__(self, path: str, fps: float, width: int, height: int):
        self.path = str(path)
        self.fps = fps
        self.width = width
        self.height = height

        # Ensure output directory exists
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        # Try codecs in priority order
        self.writer = None
        for codec, ext in self._CODEC_OPTIONS:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            output_path = str(Path(self.path).with_suffix(ext))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                self.writer = writer
                self.path = output_path
                break
            writer.release()

        if self.writer is None or not self.writer.isOpened():
            raise IOError(f"Cannot create video writer for: {self.path}")

    def write(self, frame: np.ndarray):
        """Write a single frame to the output video."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def release(self):
        """Release the video writer resource."""
        if self.writer is not None and self.writer.isOpened():
            self.writer.release()

    def __del__(self):
        self.release()
