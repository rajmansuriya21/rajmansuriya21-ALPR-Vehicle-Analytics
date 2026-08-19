"""
CSV visit report generator.

Generates a structured CSV report of all vehicle visits with columns:
Vehicle Number, Entry Time, Exit Time, Duration, Visit No., Status
"""

import csv
import logging
from pathlib import Path

from core.visit_store import VisitStore

logger = logging.getLogger(__name__)


class CsvReporter:
    """Generates CSV visit reports from the visit store."""

    HEADERS = [
        "Vehicle Number",
        "Entry Time",
        "Exit Time",
        "Duration",
        "Visit No.",
        "Status",
    ]

    def __init__(self, output_path: str):
        self.output_path = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def generate(self, visit_store: VisitStore):
        """
        Generate the CSV report from the visit store.

        Args:
            visit_store: VisitStore containing all visit records.
        """
        visits = visit_store.get_all_visits()

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)

            for visit in visits:
                writer.writerow([
                    visit.vehicle_number,
                    visit.entry_time or "—",
                    visit.exit_time or "—",
                    visit.duration or "—",
                    visit.visit_no,
                    visit.status,
                ])

        logger.info(
            f"CSV report saved: {self.output_path} ({len(visits)} visits)"
        )
