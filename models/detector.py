"""
Vehicle and license plate detection using YOLO11.

Two-stage detection pipeline:
1. VehicleDetector: Detects vehicles (car, truck, bus, motorcycle) using COCO-pretrained YOLO11.
2. PlateDetector: Detects license plates using a dedicated YOLO model (optional).

If no plate model is configured, the system falls back to PaddleOCR's
built-in text detection on vehicle crops (handled in the pipeline).
"""

import logging
from typing import List, Dict

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """
    Detects and tracks vehicles in video frames using YOLO11.

    Uses Ultralytics' built-in BoTSORT/ByteTrack tracker to maintain
    persistent track IDs across frames.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "cpu",
        confidence: float = 0.4,
        tracker_type: str = "botsort",
    ):
        logger.info(f"Loading vehicle detection model: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.tracker_type = tracker_type
        
        # Dynamically determine classes to track based on model's trained classes
        standard_vehicles = ["car", "motorcycle", "bus", "truck"]
        self.vehicle_class_ids = [k for k, v in self.model.names.items() if v.lower() in standard_vehicles]
        
        # If no standard vehicles found (e.g. it's a license plate model), track all classes
        if not self.vehicle_class_ids:
            self.vehicle_class_ids = list(self.model.names.keys())
            logger.info(f"Tracking custom model classes: {self.model.names}")

    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles and assign persistent track IDs.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of detection dicts with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
                - track_id: int or None
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=f"{self.tracker_type}.yaml",
            conf=self.confidence,
            classes=self.vehicle_class_ids,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_only(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles without tracking (single-frame mode)."""
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.vehicle_class_ids,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def _parse_results(self, results) -> List[Dict]:
        """Parse Ultralytics results into standardized detection dicts."""
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                track_id = None
                if box.id is not None:
                    track_id = int(box.id.item())

                det = {
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf.item()),
                    "class_id": int(box.cls.item()),
                    "class_name": self.model.names.get(int(box.cls.item()), "object"),
                    "track_id": track_id,
                }
                detections.append(det)
        return detections


class PlateDetector:
    """
    Detects license plates in frames using a dedicated YOLO model.

    This is optional — if no plate model is configured, the pipeline
    uses PaddleOCR's built-in text detection instead.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        confidence: float = 0.3,
    ):
        self.enabled = bool(model_path) and model_path.lower() not in ("", "none")
        self.device = device
        self.confidence = confidence

        if self.enabled:
            try:
                logger.info(f"Loading plate detection model: {model_path}")
                self.model = YOLO(model_path)
            except Exception as e:
                logger.warning(f"Failed to load plate model '{model_path}': {e}. "
                               "Falling back to PaddleOCR text detection.")
                self.enabled = False

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect license plates in a frame or crop.

        Args:
            frame: BGR image (full frame or vehicle crop).

        Returns:
            List of plate detection dicts with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
        """
        if not self.enabled:
            return []

        results = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        plates = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                plates.append({
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf.item()),
                })
        return plates

    def detect_in_vehicle(
        self, frame: np.ndarray, vehicle_bbox: list
    ) -> List[Dict]:
        """
        Detect plates within a vehicle bounding box region.

        Crops the vehicle region, runs detection, then translates
        plate coordinates back to the full frame.

        Args:
            frame: Full frame (BGR).
            vehicle_bbox: [x1, y1, x2, y2] of the vehicle.

        Returns:
            List of plate dicts with bbox in full-frame coordinates.
        """
        if not self.enabled:
            return []

        vx1, vy1, vx2, vy2 = map(int, vehicle_bbox)

        # Clamp to frame bounds
        h, w = frame.shape[:2]
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(w, vx2), min(h, vy2)

        if vx2 - vx1 < 10 or vy2 - vy1 < 10:
            return []

        crop = frame[vy1:vy2, vx1:vx2]
        plates = self.detect(crop)

        # Translate coordinates to full frame
        for plate in plates:
            plate["bbox"][0] += vx1
            plate["bbox"][1] += vy1
            plate["bbox"][2] += vx1
            plate["bbox"][3] += vy1

        return plates
