"""
Vehicle visit record management.

Manages the lifecycle of vehicle visits:
- ENTRY → Creates new visit record (status: "Inside")
- EXIT → Updates matching visit with exit time and duration (status: "Completed")
- Re-ENTRY → Creates a new visit with incremented visit_no, preserving history

Thread-safe for concurrent access from WebSocket handlers.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class Visit:
    """A single visit record for a vehicle."""
    vehicle_number: str
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    duration: Optional[str] = None
    visit_no: int = 1
    status: str = "Inside"  # "Inside" or "Completed"

    def to_dict(self) -> dict:
        return asdict(self)


class VisitStore:
    """
    In-memory store for vehicle visit records.

    Maintains a dictionary keyed by vehicle_number, where each value
    is a list of Visit records (supporting multiple visits per vehicle).
    """

    def __init__(self):
        self._visits: Dict[str, List[Visit]] = {}
        self._lock = Lock()

    def record_entry(self, vehicle_number: str, timestamp: str) -> Visit:
        """
        Record a vehicle entry event.

        If the vehicle has no active visit (or all previous visits are Completed),
        creates a new visit. Increments visit_no based on history.

        Args:
            vehicle_number: Recognized license plate number.
            timestamp: ISO format timestamp of the entry event.

        Returns:
            The created Visit record.
        """
        with self._lock:
            if vehicle_number not in self._visits:
                self._visits[vehicle_number] = []

            visits = self._visits[vehicle_number]

            # Check if there's already an active (Inside) visit
            for v in visits:
                if v.status == "Inside":
                    logger.debug(
                        f"Vehicle {vehicle_number} already has an active visit "
                        f"(#{v.visit_no}). Ignoring duplicate entry."
                    )
                    return v

            # Create new visit
            visit_no = len(visits) + 1
            visit = Visit(
                vehicle_number=vehicle_number,
                entry_time=timestamp,
                visit_no=visit_no,
                status="Inside",
            )
            visits.append(visit)

            logger.info(
                f"ENTRY: {vehicle_number} | Visit #{visit_no} | {timestamp}"
            )
            return visit

    def record_exit(self, vehicle_number: str, timestamp: str) -> Visit:
        """
        Record a vehicle exit event.

        Finds the most recent "Inside" visit for this vehicle and
        updates it with exit time and duration.

        Args:
            vehicle_number: Recognized license plate number.
            timestamp: ISO format timestamp of the exit event.

        Returns:
            The updated Visit record.
        """
        with self._lock:
            if vehicle_number not in self._visits:
                self._visits[vehicle_number] = []

            visits = self._visits[vehicle_number]

            # Find the most recent "Inside" visit
            for visit in reversed(visits):
                if visit.status == "Inside":
                    visit.exit_time = timestamp
                    visit.duration = self._calculate_duration(
                        visit.entry_time, timestamp
                    )
                    visit.status = "Completed"

                    logger.info(
                        f"EXIT: {vehicle_number} | Visit #{visit.visit_no} | "
                        f"Duration: {visit.duration} | {timestamp}"
                    )
                    return visit

            # No matching "Inside" visit — exit without entry
            visit_no = len(visits) + 1
            visit = Visit(
                vehicle_number=vehicle_number,
                exit_time=timestamp,
                visit_no=visit_no,
                status="Completed",
                duration="—",
            )
            visits.append(visit)

            logger.warning(
                f"EXIT without ENTRY: {vehicle_number} | Visit #{visit_no} | {timestamp}"
            )
            return visit

    def get_all_visits(self) -> List[Visit]:
        """Return all visit records across all vehicles, ordered by visit time."""
        with self._lock:
            all_visits = []
            for visits in self._visits.values():
                all_visits.extend(visits)

            # Sort by entry time (or exit time for orphaned exits)
            all_visits.sort(
                key=lambda v: v.entry_time or v.exit_time or ""
            )
            return all_visits

    def get_vehicle_visits(self, vehicle_number: str) -> List[Visit]:
        """Return all visits for a specific vehicle."""
        with self._lock:
            return list(self._visits.get(vehicle_number, []))

    def get_summary(self) -> dict:
        """Return summary statistics."""
        with self._lock:
            all_visits = []
            for visits in self._visits.values():
                all_visits.extend(visits)

            total_entries = sum(1 for v in all_visits if v.entry_time)
            total_exits = sum(1 for v in all_visits if v.exit_time)
            vehicles_inside = sum(1 for v in all_visits if v.status == "Inside")
            unique_vehicles = len(self._visits)

            return {
                "total_entries": total_entries,
                "total_exits": total_exits,
                "vehicles_inside": vehicles_inside,
                "unique_vehicles": unique_vehicles,
                "total_visits": len(all_visits),
            }

    @staticmethod
    def _calculate_duration(entry_time: Optional[str], exit_time: str) -> str:
        """Calculate duration between entry and exit timestamps."""
        if not entry_time:
            return "—"

        try:
            fmt = "%Y-%m-%dT%H:%M:%S"
            entry_dt = datetime.strptime(entry_time, fmt)
            exit_dt = datetime.strptime(exit_time, fmt)
            delta = exit_dt - entry_dt

            if delta.total_seconds() < 0:
                return "—"

            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except (ValueError, TypeError):
            return "—"

    def reset(self):
        """Clear all visit records."""
        with self._lock:
            self._visits.clear()
