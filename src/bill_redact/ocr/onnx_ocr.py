"""Offline OCR with PP-OCR models exported to ONNX (detection + recognition).

Pipeline per page image:

1. **Detection** (DB text detector): resize so the long side is
   ``det_limit_side`` (multiple of 32), normalise, run the model, then turn the
   probability map into quadrilateral boxes (:mod:`.db_postprocess`).
2. **Orientation**: a few of the largest boxes are cropped and the 0/90/180/270
   rotation with the best mean recognition confidence is chosen for the page;
   an optional 0/180 angle classifier (``cls.onnx``) is consulted first.
3. **Recognition** (CTC): each box is cropped, rotated, optionally deskewed,
   resized to height 48 and decoded greedily with the model's character list.

The character list is read from the recognition model's ``character``
metadata (as in the RapidOCR ONNX exports), or from a ``rec.yml`` /
``*_dict.txt`` next to the model. See ``bill-redact fetch-models``.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from ..analysis import TextBox
from .db_postprocess import DBPostProcess

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR_ENV = "BILL_REDACT_MODELS"


def default_model_dir() -> Path:
    env = os.environ.get(DEFAULT_MODEL_DIR_ENV)
    return Path(env) if env else Path("models")


def _session(path: Path, providers: Optional[Sequence[str]] = None):
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(f"onnxruntime is required for OCR: {exc}") from exc
    if not path.exists():
        raise FileNotFoundError(
            f"OCR model not found: {path}. Run `bill-redact fetch-models` or pass --models-dir.")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = list(providers or ["CPUExecutionProvider"])
    try:
        return ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    except Exception:
        return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _to_bgr(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _load_charset(model_dir: Path, sess) -> List[str]:
    meta = sess.get_modelmeta().custom_metadata_map or {}
    if meta.get("character"):
        chars = meta["character"].splitlines()
    else:
        chars = []
        yml = model_dir / "rec.yml"
        if yml.exists():
            import yaml
            with open(yml, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            chars = [str(c) for c in ((cfg.get("PostProcess") or {}).get("character_dict") or [])]
        if not chars:
            dicts = sorted(model_dir.glob("*dict*.txt"))
            if not dicts:
                raise FileNotFoundError(
                    "recognition model has no embedded character list and no rec.yml / *_dict.txt was found")
            chars = dicts[0].read_text(encoding="utf-8").splitlines()
    # CTC: index 0 is blank; PaddleOCR appends a space as the last class.
    return ["<blank>"] + chars + [" "]


class OnnxOcr:
    """PP-OCR detection + recognition with ONNX Runtime."""

    def __init__(self, model_dir: Optional[str | Path] = None, *, det_limit_side: int = 960,
                 det_thresh: float = 0.3, box_thresh: float = 0.6, unclip_ratio: float = 1.5,
                 rec_height: int = 48, deskew: bool = True, max_deskew_degrees: float = 12.0,
                 try_right_angles: bool = True, providers: Optional[Sequence[str]] = None):
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.det = _session(self.model_dir / "det.onnx", providers)
        self.rec = _session(self.model_dir / "rec.onnx", providers)
        cls_path = self.model_dir / "cls.onnx"
        self.cls = _session(cls_path, providers) if cls_path.exists() else None
        self.chars = _load_charset(self.model_dir, self.rec)
        self.det_limit_side = det_limit_side
        self.post = DBPostProcess(thresh=det_thresh, box_thresh=box_thresh,
                                  unclip_ratio=unclip_ratio, max_candidates=1000)
        self.rec_height = rec_height
        self.deskew = deskew
        self.max_deskew_degrees = max_deskew_degrees
        self.try_right_angles = try_right_angles

    # ---------------------------------------------------------------- detection
    def detect(self, img_bgr: np.ndarray) -> List[np.ndarray]:
        h, w = img_bgr.shape[:2]
        ratio = self.det_limit_side / float(max(h, w))
        nh = max(32, int(round(h * ratio / 32.0)) * 32)
        nw = max(32, int(round(w * ratio / 32.0)) * 32)
        x = cv2.resize(img_bgr, (nw, nh)).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None]
        pred = self.det.run(None, {self.det.get_inputs()[0].name: x})[0]
        out = self.post({"maps": pred}, [[h, w, nh / h, nw / w]])
        boxes = [np.asarray(b, dtype=np.int32) for b in out[0]["points"]]
        # reading order: top to bottom, then left to right
        boxes.sort(key=lambda b: (int(b[:, 1].min() / 10), b[:, 0].min()))
        return boxes

    # -------------------------------------------------------------- recognition
    def _rec_tensor(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((1, 3, self.rec_height, 8), np.float32)
        rw = max(1, min(int(math.ceil(self.rec_height * w / float(h))), 3200))
        x = cv2.resize(crop, (rw, self.rec_height)).astype(np.float32)
        x = (x.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        padded = np.zeros((3, self.rec_height, rw * 2), np.float32)
        padded[:, :, :rw] = x
        return padded[None]

    def recognize(self, crop: np.ndarray) -> Tuple[str, float]:
        x = self._rec_tensor(_to_bgr(crop))
        preds = self.rec.run(None, {self.rec.get_inputs()[0].name: x})[0]
        if preds.ndim == 3 and preds.shape[1] > preds.shape[2]:
            preds = preds.transpose(0, 2, 1)
        idx = preds[0].argmax(axis=1)
        prob = preds[0].max(axis=1)
        chars, confs, prev = [], [], -1
        for j, k in enumerate(idx):
            k = int(k)
            if k != 0 and k != prev and k < len(self.chars):
                chars.append(self.chars[k])
                confs.append(float(prob[j]))
            prev = k
        return "".join(chars), float(np.mean(confs)) if confs else 0.0

    def _cls_angle(self, crops: List[np.ndarray]) -> float:
        """0 or 180 from the optional angle classifier (most confident crop wins)."""
        if self.cls is None or not crops:
            return 0.0
        batch = []
        for c in crops:
            h, w = c.shape[:2]
            rw = min(192, int(math.ceil(48 * w / float(max(h, 1)))))
            x = cv2.resize(c, (max(rw, 1), 48)).astype(np.float32).transpose(2, 0, 1) / 255.0
            x = (x - 0.5) / 0.5
            p = np.zeros((3, 48, 192), np.float32)
            p[:, :, :x.shape[2]] = x
            batch.append(p)
        probs = self.cls.run(None, {self.cls.get_inputs()[0].name: np.stack(batch)})[0]
        best = int(np.argmax(probs.max(axis=1)))
        k = int(np.argmax(probs[best]))
        return 180.0 if (k == 1 and probs[best, k] > 0.9) else 0.0

    @staticmethod
    def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Small-angle deskew of a crop using Hough lines (no-op when level)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(bw) < 127:
            bw = 255 - bw
        edges = cv2.Canny(bw, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=max(80, int(min(img.shape[:2]) * 0.15)))
        if lines is None:
            return img
        angles = []
        for (rho, theta), in lines[:200]:
            a = theta * 180.0 / math.pi - 90.0
            if -45.0 <= a <= 45.0:
                angles.append(a)
        if not angles:
            return img
        angle = float(np.clip(np.median(angles), -self.max_deskew_degrees, self.max_deskew_degrees))
        if abs(angle) < 0.5:
            return img
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------- public
    def __call__(self, image) -> List[TextBox]:
        img = _to_bgr(image)
        boxes = self.detect(img)
        if not boxes:
            return []

        def crop(box: np.ndarray) -> np.ndarray:
            x, y, w, h = cv2.boundingRect(box)
            return img[max(0, y):y + h, max(0, x):x + w]

        crops = [crop(b) for b in boxes]
        # page orientation from the five largest boxes
        order = np.argsort([c.shape[0] * c.shape[1] for c in crops])[-5:]
        sample = [crops[i] for i in order if crops[i].size]
        angle = int(self._cls_angle(sample))
        if self.try_right_angles and sample:
            best, best_conf = angle, -1.0
            for cand in (0, 90, 180, 270):
                conf = float(np.mean([self.recognize(self._rotate(c, cand))[1] for c in sample]))
                if conf > best_conf:
                    best, best_conf = cand, conf
            angle = best

        results: List[TextBox] = []
        for box, c in zip(boxes, crops):
            if c.size == 0:
                continue
            c = self._rotate(c, angle)
            if self.deskew:
                c = self._deskew(c)
            text, conf = self.recognize(c)
            quad = [(float(px), float(py)) for px, py in box.tolist()]
            results.append(TextBox(text, quad, conf))
        return results
