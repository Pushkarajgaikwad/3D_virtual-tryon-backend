"""
FaceExtractor — detects face landmarks and extracts face patches using MediaPipe.
"""

import logging
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FaceExtractor:
    """
    Detects facial landmarks and extracts aligned face patches from images.

    Uses MediaPipe FaceMesh when available. Falls back gracefully if MediaPipe
    is not installed.

    Args:
        model_selection: 0 = short-range (≤2 m), 1 = full-range.
    """

    def __init__(self, model_selection: int = 0):
        self._mp_available = False
        self._face_mesh = None

        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                model_selection=model_selection,
            )
            self._mp_available = True
            logger.info("FaceExtractor: MediaPipe FaceMesh loaded")
        except ImportError:
            logger.warning(
                "FaceExtractor: mediapipe not installed. "
                "Face landmark detection disabled. "
                "Install with: pip install mediapipe"
            )
        except Exception as e:
            logger.warning(f"FaceExtractor: Failed to initialise MediaPipe: {e}")

    # ------------------------------------------------------------------

    def detect_landmarks(self, image_bytes: bytes) -> List[Tuple[float, float]]:
        """
        Detect 2-D face landmarks.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            List of (x, y) tuples in pixel coordinates.
            Returns empty list if no face detected or MediaPipe unavailable.
        """
        if not self._mp_available:
            return []

        import cv2
        img_array = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
        results = self._face_mesh.process(img_array)

        if not results.multi_face_landmarks:
            logger.debug("No face detected in image")
            return []

        h, w = img_array.shape[:2]
        landmarks = results.multi_face_landmarks[0]
        return [(lm.x * w, lm.y * h) for lm in landmarks.landmark]

    def extract_face_patch(
        self,
        image_bytes: bytes,
        padding: float = 0.2,
    ) -> Tuple[Optional[Image.Image], List[Tuple[float, float]], Tuple[int, int, int, int]]:
        """
        Extract a cropped face patch from an image.

        Args:
            image_bytes: Raw image bytes.
            padding:     Fractional padding around detected face bounding box.

        Returns:
            face_patch      – PIL.Image of the cropped face, or None if no face.
            landmarks_norm  – Landmarks normalised to [0, 1] within the crop.
            bbox            – (x0, y0, x1, y1) bounding box in original pixels.
        """
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        landmarks = self.detect_landmarks(image_bytes)
        if not landmarks:
            # Return centre crop as fallback
            cx, cy = w // 2, h // 4
            half = min(w, h) // 4
            bbox = (max(0, cx - half), max(0, cy - half),
                    min(w, cx + half), min(h, cy + half))
            face_patch = img.crop(bbox)
            return face_patch, [], bbox

        xs = [pt[0] for pt in landmarks]
        ys = [pt[1] for pt in landmarks]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

        pad_x = (x1 - x0) * padding
        pad_y = (y1 - y0) * padding
        x0 = max(0, int(x0 - pad_x))
        y0 = max(0, int(y0 - pad_y))
        x1 = min(w, int(x1 + pad_x))
        y1 = min(h, int(y1 + pad_y))
        bbox = (x0, y0, x1, y1)

        face_patch = img.crop(bbox)
        crop_w = x1 - x0
        crop_h = y1 - y0

        landmarks_norm = [
            ((pt[0] - x0) / max(crop_w, 1), (pt[1] - y0) / max(crop_h, 1))
            for pt in landmarks
        ]

        return face_patch, landmarks_norm, bbox
