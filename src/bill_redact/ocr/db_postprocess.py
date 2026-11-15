"""DB (Differentiable Binarization) post-processing for the text detector.

Turns the detector's probability map into quadrilateral (or polygon) text
boxes in original image coordinates. Adapted from the DBPostProcess in
PaddleOCR (Apache-2.0).
"""

from __future__ import annotations

import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper
from typing import List, Tuple, Dict, Any


class DBPostProcess:
    """Post process for Differentiable Binarization (DB).

    Parameters mirror common YAML config fields.
    """

    def __init__(self,
                 thresh: float = 0.3,
                 box_thresh: float = 0.7,
                 max_candidates: int = 1000,
                 unclip_ratio: float = 2.0,
                 use_dilation: bool = False,
                 score_mode: str = "fast",
                 box_type: str = 'quad',
                 **_: Dict[str, Any]) -> None:
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.max_candidates = int(max_candidates)
        self.unclip_ratio = float(unclip_ratio)
        self.min_size = 3
        self.score_mode = str(score_mode)
        self.box_type = str(box_type)
        assert self.score_mode in {"slow", "fast"}

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def polygons_from_bitmap(self,
                              pred: np.ndarray,
                              _bitmap: np.ndarray,
                              dest_width: int,
                              dest_height: int) -> Tuple[List[List[List[float]]], List[float]]:
        bitmap = _bitmap
        height, width = bitmap.shape

        boxes: List[List[List[float]]] = []
        scores: List[float] = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[: self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = float(self.box_score_fast(pred, points.reshape(-1, 2)))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self,
                           pred: np.ndarray,
                           _bitmap: np.ndarray,
                           dest_width: int,
                           dest_height: int) -> Tuple[np.ndarray, List[float]]:
        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _, contours, _ = outs
        else:
            contours, _ = outs

        num_contours = min(len(contours), self.max_candidates)

        boxes: List[np.ndarray] = []
        scores: List[float] = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = float(self.box_score_fast(pred, points.reshape(-1, 2)))
            else:
                score = float(self.box_score_slow(pred, contour))
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box: np.ndarray, unclip_ratio: float) -> np.ndarray:
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour: np.ndarray) -> Tuple[List[List[float]], float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, float(min(bounding_box[1]))

    def box_score_fast(self, bitmap: np.ndarray, _box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])

    def box_score_slow(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])

    def __call__(self, outs_dict: Dict[str, np.ndarray], shape_list: List[List[float]]):
        pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch: List[Dict[str, Any]] = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            if self.box_type == 'poly':
                boxes, _ = self.polygons_from_bitmap(pred[batch_index], mask, int(src_w), int(src_h))
            elif self.box_type == 'quad':
                boxes, _ = self.boxes_from_bitmap(pred[batch_index], mask, int(src_w), int(src_h))
            else:
                raise ValueError("box_type can only be one of ['quad', 'poly']")

            boxes_batch.append({'points': boxes})
        return boxes_batch


__all__ = ["DBPostProcess"]
