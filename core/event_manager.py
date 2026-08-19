"""
Event management for entry/exit detection.

Receives line-crossing signals from the tracker, resolves plate numbers
via the aggregator, creates structured event records, and delegates
visit lifecycle updates to the visit store.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from core.visit_store import VisitStore

logger = logging.getLogger(__name__)


@dataclass
class VehicleEvent:
    """A single detected vehicle event (ENTRY or EXIT)."""
    vehicle_number: str
    event: str           # "ENTRY" or "EXIT"
    timestamp: str       # ISO format: "2026-08-09T09:15:20"
    camera: str          # Camera identifier
    track_id: int = -1   # Internal track ID (for debugging)

    def to_dict(self) -> dict:
        return {
            "vehicle_number": self.vehicle_number,
            "event": self.event,
            "timestamp": self.timestamp,
            "camera": self.camera,
        }


class EventManager:
    """
    Creates and manages vehicle entry/exit events.

    Coordinates between the line crossing detector, plate aggregator,
    and visit store to produce structured event records.
    """

    def __init__(
        self,
        visit_store: VisitStore,
        camera_id: str = "camera_1",
        on_event: Optional[Callable[[VehicleEvent], None]] = None,
    ):
        """
        Args:
            visit_store: Visit record store instance.
            camera_id: Camera identifier for event logs.
            on_event: Optional callback invoked on each new event.
        """
        self.visit_store = visit_store
        self.camera_id = camera_id
        self.on_event = on_event
        self._events: List[VehicleEvent] = []

    def create_event(
        self,
        vehicle_number: str,
        event_type: str,
        timestamp: str,
        track_id: int = -1,
    ) -> VehicleEvent:
        """
        Create a new vehicle event and update the visit store.

        Args:
            vehicle_number: Recognized license plate number.
            event_type: "ENTRY" or "EXIT".
            timestamp: ISO format timestamp.
            track_id: Internal track ID.

        Returns:
            The created VehicleEvent.
        """
        event = VehicleEvent(
            vehicle_number=vehicle_number,
            event=event_type,
            timestamp=timestamp,
            camera=self.camera_id,
            track_id=track_id,
        )

        # Update visit store
        if event_type == "ENTRY":
            self.visit_store.record_entry(vehicle_number, timestamp)
        elif event_type == "EXIT":
            self.visit_store.record_exit(vehicle_number, timestamp)

        # Store event
        self._events.append(event)

        # Invoke callback (for WebSocket streaming, etc.)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        logger.info(
            f"Event: {event_type} | Vehicle: {vehicle_number} | "
            f"Time: {timestamp} | Track: {track_id}"
        )

        return event

    def get_all_events(self) -> List[Dict]:
        """Return all events as list of dicts."""
        return [e.to_dict() for e in self._events]

    def get_events_count(self) -> Dict[str, int]:
        """Return count of entry and exit events."""
        entries = sum(1 for e in self._events if e.event == "ENTRY")
        exits = sum(1 for e in self._events if e.event == "EXIT")
        return {"entries": entries, "exits": exits, "total": len(self._events)}

    @staticmethod
    def frame_to_timestamp(frame_idx: int, fps: float, start_time: Optional[datetime] = None) -> str:
        """
        Convert frame index to ISO timestamp string.

        Args:
            frame_idx: Frame number (0-indexed).
            fps: Video frames per second.
            start_time: Video start time (defaults to current time).

        Returns:
            ISO format timestamp string.
        """
        if start_time is None:
            start_time = datetime.now()

        seconds_offset = frame_idx / fps
        event_time = start_time + timedelta(seconds=seconds_offset)
        return event_time.strftime("%Y-%m-%dT%H:%M:%S")

    def reset(self):
        """Clear all events."""
        self._events.clear()
