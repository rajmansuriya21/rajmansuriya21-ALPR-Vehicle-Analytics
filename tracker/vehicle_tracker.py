"""
Vehicle tracker wrapper.

Wraps the Ultralytics built-in BoTSORT/ByteTrack tracker and provides
a clean interface for the pipeline. Maintains centroid history per track_id.
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from utils.geometry import centroid_from_bbox

logger = logging.getLogger(__name__)

# Number of past centroid positions to keep per track
HISTORY_LENGTH = 30


class VehicleTracker:
    """
    Manages vehicle tracking state and centroid history.

    The actual tracking is done by Ultralytics YOLO (BoTSORT/ByteTrack).
    This class maintains additional state on top:
    - Centroid history per track_id
    - Track lifecycle (new, active, lost)
    """

    def __init__(self, history_length: int = HISTORY_LENGTH):
        self.history_length = history_length
        # Centroid history: track_id -> list of (x, y)
        self._centroid_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        # Track status
        self._active_tracks: set = set()
        # Frame when track was last seen
        self._last_seen: Dict[int, int] = {}

    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """
        Update tracker state with new detections.

        Computes centroids and maintains position history.

        Args:
            detections: List of detection dicts from VehicleDetector.
            frame_idx: Current frame index.

        Returns:
            Enriched detections with added 'centroid' and 'is_new' keys.
        """
        current_track_ids = set()

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            # Compute centroid
            centroid = centroid_from_bbox(det["bbox"])
            det["centroid"] = centroid

            # Update history
            history = self._centroid_history[track_id]
            history.append(centroid)
            if len(history) > self.history_length:
                history.pop(0)

            # Track lifecycle
            det["is_new"] = track_id not in self._active_tracks
            self._active_tracks.add(track_id)
            self._last_seen[track_id] = frame_idx
            current_track_ids.add(track_id)

        # Clean up lost tracks (not seen for 60 frames)
        lost_threshold = frame_idx - 60
        lost_ids = [
            tid for tid, last in self._last_seen.items()
            if last < lost_threshold and tid not in current_track_ids
        ]
        for tid in lost_ids:
            self._active_tracks.discard(tid)
            self._centroid_history.pop(tid, None)
            self._last_seen.pop(tid, None)

        return detections

    def get_centroid_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Get the centroid history for a track."""
        return self._centroid_history.get(track_id, [])

    def get_previous_centroid(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Get the previous centroid for a track (second-to-last in history)."""
        history = self._centroid_history.get(track_id, [])
        if len(history) >= 2:
            return history[-2]
        return None

    def get_active_track_ids(self) -> set:
        """Return the set of currently active track IDs."""
        return self._active_tracks.copy()

    def reset(self):
        """Reset all tracking state."""
        self._centroid_history.clear()
        self._active_tracks.clear()
        self._last_seen.clear()
