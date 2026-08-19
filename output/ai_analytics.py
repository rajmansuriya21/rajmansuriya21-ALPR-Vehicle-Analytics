"""
AI-powered analytics report generator.

Uses an LLM (Google Gemini or OpenAI) to analyze the structured JSON
event logs and generate a comprehensive analytics report including:
- Executive Summary
- Missing Entry/Exit Records
- Unusual Activity Detection
- Recommendations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

ANALYTICS_PROMPT = """You are a security analytics expert analyzing vehicle entry/exit logs from a CCTV monitoring system.

Based on the following structured event data, generate a comprehensive analytics report.

## Event Log Data
{events_json}

## Summary Statistics
{summary_json}

## Required Report Sections

### 1. Executive Summary
Provide a concise overview of the monitoring period including:
- Total number of entries and exits
- Number of unique vehicles
- Peak activity periods (if observable from timestamps)
- Overall compliance rate (matched entries/exits)

### 2. Missing Entry/Exit Records
Identify and list:
- Vehicles that have an entry but no corresponding exit (still inside or missing data)
- Vehicles that have an exit but no corresponding entry (data gap)
- Assessment of data completeness

### 3. Unusual Activity Detection
Flag any suspicious or noteworthy patterns:
- Vehicles with unusually short visits (potential turnaround)
- Vehicles with very long visits (potential overstay)
- Frequent visitors (multiple visits)
- Any temporal anomalies

### 4. Recommendations
Based on the observed patterns, provide actionable recommendations for:
- Improving gate monitoring
- Addressing data gaps
- Security considerations
- Operational improvements

Format the report in clean Markdown with clear headings and bullet points.
Be specific with vehicle numbers and timestamps when citing evidence.
"""


class AiAnalytics:
    """Generates AI-powered analytics reports from event logs."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: str = "",
        model: str = "gemini-2.0-flash",
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model

    def generate_report(
        self,
        events: List[Dict],
        summary: Dict,
        output_path: str,
    ) -> str:
        """
        Generate an AI analytics report from event data.

        Args:
            events: List of event dicts from the JSON logger.
            summary: Summary statistics from the visit store.
            output_path: Path to save the markdown report.

        Returns:
            Generated report text.
        """
        if not events:
            report = self._generate_empty_report()
        else:
            prompt = ANALYTICS_PROMPT.format(
                events_json=json.dumps(events, indent=2),
                summary_json=json.dumps(summary, indent=2),
            )

            try:
                if self.provider == "gemini":
                    report = self._call_gemini(prompt)
                elif self.provider == "openai":
                    report = self._call_openai(prompt)
                else:
                    logger.error(f"Unknown LLM provider: {self.provider}")
                    report = self._generate_fallback_report(events, summary)
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                report = self._generate_fallback_report(events, summary)

        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"AI analytics report saved: {output_path}")
        return report

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return response.text

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _generate_empty_report(self) -> str:
        """Generate report when no events were detected."""
        return """# Vehicle Analytics Report

## Executive Summary
No vehicle entry or exit events were detected during the monitoring period.

## Possible Reasons
- No vehicles passed through the monitored area
- Detection lines may need adjustment
- Camera angle or video quality may be insufficient
- Vehicle detection confidence threshold may be too high

## Recommendations
- Verify the entry/exit line positions are correctly configured
- Review the video to confirm vehicles are visible and plates are readable
- Consider lowering the confidence threshold in the .env configuration
"""

    def _generate_fallback_report(self, events: List[Dict], summary: Dict) -> str:
        """Generate a basic report without LLM (fallback when API fails)."""
        report = "# Vehicle Analytics Report\n\n"
        report += "*(Generated without AI — LLM API unavailable)*\n\n"

        report += "## Executive Summary\n"
        report += f"- **Total Events:** {len(events)}\n"
        report += f"- **Total Entries:** {summary.get('total_entries', 0)}\n"
        report += f"- **Total Exits:** {summary.get('total_exits', 0)}\n"
        report += f"- **Unique Vehicles:** {summary.get('unique_vehicles', 0)}\n"
        report += f"- **Vehicles Currently Inside:** {summary.get('vehicles_inside', 0)}\n\n"

        report += "## Event Log\n"
        report += "| Vehicle | Event | Timestamp |\n"
        report += "|---------|-------|-----------|\n"
        for e in events:
            report += f"| {e.get('vehicle_number', '?')} | {e.get('event', '?')} | {e.get('timestamp', '?')} |\n"

        return report
