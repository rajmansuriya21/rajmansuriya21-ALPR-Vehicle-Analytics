"""
License plate OCR using PaddleOCR.

Handles both:
1. Reading text from pre-cropped plate images (when plate detector is available).
2. Full detection + recognition on vehicle crops (fallback mode).

Includes image pre-processing (CLAHE, resize, grayscale) and post-processing
(text normalization, Indian plate format validation).
"""

import logging
import re
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Indian license plate patterns (various formats)
INDIAN_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$"),      # KA01AB1234
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$"),     # KA01A1234
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$"),   # Flexible
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),              # BH series
]


class PlateOCR:
    """
    License plate text recognition using PaddleOCR.

    Supports two modes:
    - read_plate(): Reads text from a pre-cropped plate image.
    - detect_and_read(): Detects text regions AND reads them in a vehicle crop.
    """

    def __init__(self, lang: str = "en", confidence_threshold: float = 0.5):
        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR engine...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            use_gpu=False,
        )
        self.confidence_threshold = confidence_threshold

    def read_plate(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Read text from a pre-cropped license plate image.

        Args:
            plate_image: Cropped plate region (BGR or grayscale).

        Returns:
            (plate_text, confidence) or (None, 0.0) if nothing detected.
        """
        if plate_image is None or plate_image.size == 0:
            return None, 0.0

        processed = self._preprocess(plate_image)

        try:
            results = self.ocr.ocr(processed, cls=True)
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return None, 0.0

        if not results or not results[0]:
            return None, 0.0

        # Combine all detected text lines and find the best plate-like text
        best_text = ""
        best_conf = 0.0

        for line in results[0]:
            text, conf = line[1]
            text = self._postprocess(text)

            if conf > best_conf and len(text) >= 4:
                best_text = text
                best_conf = conf

        if best_conf < self.confidence_threshold or not best_text:
            return None, best_conf

        return best_text, best_conf

    def detect_and_read(self, vehicle_crop: np.ndarray) -> List[Dict]:
        """
        Run full OCR pipeline (detection + recognition) on a vehicle crop.

        Used as fallback when no dedicated plate detection model is available.
        PaddleOCR's text detector finds text regions, then the recognizer
        reads them. Results are filtered by plate-like patterns.

        Args:
            vehicle_crop: Cropped vehicle region (BGR).

        Returns:
            List of detected plates with keys: text, confidence, bbox_in_crop.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        # Ensure minimum size for OCR
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return []

        try:
            results = self.ocr.ocr(vehicle_crop, cls=True)
        except Exception as e:
            logger.debug(f"OCR detection error: {e}")
            return []

        if not results or not results[0]:
            return []

        plates = []
        for line in results[0]:
            bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text, conf = line[1]
            text = self._postprocess(text)

            if conf >= self.confidence_threshold and self._looks_like_plate(text):
                # Convert polygon bbox to rectangular [x1, y1, x2, y2]
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                plates.append({
                    "text": text,
                    "confidence": conf,
                    "bbox_in_crop": [min(xs), min(ys), max(xs), max(ys)],
                })

        return plates

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Pre-process plate image for better OCR accuracy.

        Steps:
        1. Convert to grayscale
        2. Resize to minimum height (64px)
        3. Apply CLAHE for contrast enhancement
        """
        # Convert to grayscale if color
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small
        h, w = gray.shape[:2]
        if h < 64:
            scale = 64.0 / h
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def _postprocess(self, text: str) -> str:
        """
        Normalize OCR output for license plate format.

        Removes non-alphanumeric characters, converts to uppercase.
        """
        text = text.upper().strip()
        # Remove all non-alphanumeric characters
        text = re.sub(r"[^A-Z0-9]", "", text)
        # Common OCR substitutions
        text = text.replace("O", "0") if text[:2].isdigit() else text
        return text

    def _looks_like_plate(self, text: str) -> bool:
        """
        Heuristic check if text resembles a license plate number.

        Checks for:
        - Length between 4 and 12 characters
        - Contains both letters and digits
        - Optionally matches Indian plate patterns
        """
        if len(text) < 4 or len(text) > 12:
            return False

        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)

        if not (has_letters and has_digits):
            return False

        # Check against known Indian plate patterns
        for pattern in INDIAN_PLATE_PATTERNS:
            if pattern.match(text):
                return True

        # Flexible check: at least 2 letters and 2 digits
        letter_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        return letter_count >= 2 and digit_count >= 2
