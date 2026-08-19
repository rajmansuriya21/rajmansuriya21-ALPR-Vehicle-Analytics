"""
Vehicle Entry & Exit Analytics System — CLI Entry Point.

Processes CCTV video footage to detect vehicle license plates,
track entry/exit events across configurable virtual lines, and
generate structured reports with AI-powered analytics.

Usage:
    python main.py

All configuration is read from the .env file.
No source code changes are required to execute.
"""

import sys
import logging
from pathlib import Path

from config.settings import load_settings
from core.pipeline import VideoPipeline


def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("paddleocr").setLevel(logging.WARNING)
    logging.getLogger("ppocr").setLevel(logging.WARNING)


def print_banner():
    """Print application banner."""
    print("\n" + "=" * 60)
    print("  🚗  Vehicle Entry & Exit Analytics System")
    print("  AI-Powered CCTV License Plate Recognition")
    print("=" * 60)
    print()


def main():
    """Main entry point for CLI processing."""
    setup_logging()
    print_banner()
    logger = logging.getLogger("main")

    # 1. Load configuration
    try:
        settings = load_settings()
        logger.info("Configuration loaded from .env")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.error("Ensure .env file exists. Copy from .env.example if needed.")
        sys.exit(1)

    # 2. Ensure output directory
    output_dir = settings.ensure_output_dir()
    logger.info(f"Output directory: {output_dir}")

    # 3. Get video paths
    try:
        video_paths = settings.get_video_paths()
        logger.info(f"Videos to process: {len(video_paths)}")
        for vp in video_paths:
            logger.info(f"  → {vp}")
    except FileNotFoundError as e:
        logger.error(f"Video not found: {e}")
        sys.exit(1)

    # 4. Initialize pipeline (reads .env detection coords — horizontal, tuned for this camera)
    def on_progress(current, total, percent):
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  Processing: [{bar}] {percent:.1f}% ({current}/{total})", end="", flush=True)

    pipeline = VideoPipeline(
        settings=settings,
        on_progress=on_progress,
    )

    # 5. Interactive line drawing (if enabled)
    # The drawn line is used ONLY for visual display on-screen.
    # Detection always uses the .env coordinates (horizontal, tuned for this camera).
    if settings.interactive_line_draw:
        logger.info("Interactive line drawing mode enabled.")
        try:
            from utils.line_drawer import InteractiveLineDrawer
            from utils.video_io import VideoReader

            # Use first video's first frame
            reader = VideoReader(video_paths[0])
            first_frame = reader.get_first_frame()
            reader.release()

            if first_frame is not None:
                drawer = InteractiveLineDrawer(first_frame, settings.line_mode)
                result = drawer.draw()

                if result is None:
                    logger.warning("Line drawing cancelled. Showing .env line coordinates.")
                else:
                    logger.info(f"Lines drawn — updating visual display only: {result}")
                    # Update ONLY the visual display (not detection) with drawn coordinates
                    pipeline.set_display_lines(
                        entry_line=result.get("entry_line"),
                        exit_line=result.get("exit_line"),
                        single_line=result.get("single_line"),
                    )
            else:
                logger.warning("Could not read first frame. Using .env coordinates.")
        except Exception as e:
            logger.warning(f"Interactive line drawing failed: {e}. Using .env coordinates.")

    # 6. Process each video
    all_results = []
    for i, video_path in enumerate(video_paths):
        print(f"\n{'─' * 60}")
        logger.info(f"Processing video {i + 1}/{len(video_paths)}: {video_path}")
        print()

        try:
            results = pipeline.process_video(video_path)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Reset pipeline state for next video
        if i < len(video_paths) - 1:
            pipeline.reset()

    # 7. Print summary
    print(f"\n\n{'=' * 60}")
    print("  PROCESSING COMPLETE")
    print(f"{'=' * 60}\n")

    for result in all_results:
        summary = result["summary"]
        print(f"  Video: {Path(result['video_path']).name}")
        print(f"  ├─ Total Entries:      {summary['total_entries']}")
        print(f"  ├─ Total Exits:        {summary['total_exits']}")
        print(f"  ├─ Vehicles Inside:    {summary['vehicles_inside']}")
        print(f"  ├─ Unique Vehicles:    {summary['unique_vehicles']}")
        print(f"  └─ Total Visits:       {summary['total_visits']}")
        print()

        paths = result["output_paths"]
        print("  Output Files:")
        for key, path in paths.items():
            if path:
                print(f"    → {key}: {path}")
        print()

    print(f"{'=' * 60}\n")

    # 8. Print web dashboard hint
    print("  💡 For the web dashboard, run:")
    print("     python -m app.server")
    print(f"     Then open http://localhost:{settings.port}\n")


if __name__ == "__main__":
    main()
