"""
JSON event logger.

Writes vehicle entry/exit events to a structured JSON file.
Supports real-time appending during processing and final save.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from threading import Lock

logger = logging.getLogger(__name__)


class JsonLogger:
    """
    Logs vehicle events to a JSON file.

    Events are accumulated in memory and written to disk on save().
    Thread-safe for concurrent access.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self._events: List[Dict] = []
        self._lock = Lock()

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: Dict):
        """
        Add a new event to the log.

        Args:
            event: Event dict with keys: vehicle_number, event, timestamp, camera.
        """
        with self._lock:
            self._events.append(event)

        # Also print to console for real-time visibility
        logger.info(
            f"[EVENT] {event.get('event', '?')} | "
            f"{event.get('vehicle_number', '?')} | "
            f"{event.get('timestamp', '?')} | "
            f"{event.get('camera', '?')}"
        )

    def save(self):
        """Write all accumulated events to the JSON file."""
        with self._lock:
            events_copy = list(self._events)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(events_copy, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON log saved: {self.output_path} ({len(events_copy)} events)")

    def get_events(self) -> List[Dict]:
        """Return all logged events."""
        with self._lock:
            return list(self._events)

    def get_events_count(self) -> int:
        """Return the number of logged events."""
        with self._lock:
            return len(self._events)
