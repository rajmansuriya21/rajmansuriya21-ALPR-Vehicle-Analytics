"""
License plate text aggregation using weighted majority voting.

OCR results from individual frames are noisy. This module collects
all readings for each tracked vehicle (identified by track_id) and
uses confidence-weighted majority voting to determine the most likely
plate number.

A minimum number of consistent reads is required before committing
a plate number, ensuring reliability.
"""

import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum number of consistent OCR reads before committing a plate
MIN_CONSISTENT_READS = 2
# Maximum readings to store per track (memory limit)
MAX_READINGS_PER_TRACK = 50


class PlateAggregator:
    """
    Aggregates OCR readings per vehicle track and resolves plate numbers
    via weighted majority voting.
    """

    def __init__(self, min_reads: int = MIN_CONSISTENT_READS):
        self.min_reads = min_reads
        # track_id -> list of (plate_text, confidence)
        self._readings: Dict[int, list] = defaultdict(list)
        # track_id -> committed plate number (cached result)
        self._committed: Dict[int, str] = {}

    def add_reading(self, track_id: int, plate_text: str, confidence: float):
        """
        Add an OCR reading for a tracked vehicle.

        Args:
            track_id: Vehicle track ID.
            plate_text: Recognized plate text (already normalized).
            confidence: OCR confidence score (0-1).
        """
        if not plate_text or len(plate_text) < 4:
            return

        readings = self._readings[track_id]
        readings.append((plate_text, confidence))

        # Limit memory usage
        if len(readings) > MAX_READINGS_PER_TRACK:
            readings.pop(0)

        # Invalidate cached result
        self._committed.pop(track_id, None)

    def get_plate(self, track_id: int) -> Optional[str]:
        """
        Get the resolved plate number for a track_id.

        Uses confidence-weighted majority voting across all readings.
        Returns None if minimum consistent reads threshold is not met.

        Args:
            track_id: Vehicle track ID.

        Returns:
            Resolved plate number or None.
        """
        # Return cached result if available
        if track_id in self._committed:
            return self._committed[track_id]

        readings = self._readings.get(track_id, [])
        if not readings:
            return None

        # Weighted majority voting
        vote_weights: Dict[str, float] = defaultdict(float)
        vote_counts: Dict[str, int] = defaultdict(int)

        for text, conf in readings:
            vote_weights[text] += conf
            vote_counts[text] += 1

        if not vote_weights:
            return None

        # Find the candidate with the highest weighted vote
        best_plate = max(vote_weights, key=vote_weights.get)
        best_count = vote_counts[best_plate]

        # Require minimum consistent reads
        if best_count < self.min_reads:
            return None

        # Cache and return
        self._committed[track_id] = best_plate
        logger.debug(
            f"Track {track_id}: Committed plate '{best_plate}' "
            f"({best_count} reads, weight={vote_weights[best_plate]:.2f})"
        )
        return best_plate

    def get_plate_with_confidence(
        self, track_id: int
    ) -> Tuple[Optional[str], float]:
        """
        Get resolved plate number with average confidence.

        Returns:
            (plate_text, avg_confidence) or (None, 0.0).
        """
        plate = self.get_plate(track_id)
        if plate is None:
            return None, 0.0

        readings = self._readings.get(track_id, [])
        matching = [conf for text, conf in readings if text == plate]
        avg_conf = sum(matching) / len(matching) if matching else 0.0

        return plate, avg_conf

    def get_all_readings(self, track_id: int) -> list:
        """Return all raw readings for a track_id."""
        return list(self._readings.get(track_id, []))

    def reset(self):
        """Clear all aggregation state."""
        self._readings.clear()
        self._committed.clear()
