"""Download the public PP-OCR ONNX models used for scanned pages.

The models are the stock PaddleOCR PP-OCRv5 detector/recogniser and the
PP-OCR 0/180 angle classifier, in the ONNX exports published by the RapidOCR
project. Nothing here is trained on private documents. Checksums are verified
after download.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx"

# name -> (url, sha256)
MODELS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "mobile": {
        "det.onnx": (f"{_BASE}/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
                     "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"),
        "rec.onnx": (f"{_BASE}/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
                     "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5"),
        "cls.onnx": (f"{_BASE}/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                     "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
    },
    "server": {
        "det.onnx": (f"{_BASE}/PP-OCRv5/det/ch_PP-OCRv5_det_server.onnx",
                     "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad"),
        "rec.onnx": (f"{_BASE}/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
                     "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a"),
        "cls.onnx": (f"{_BASE}/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                     "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_models(dest: str | Path = "models", variant: str = "mobile", force: bool = False) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in MODELS[variant].items():
        target = dest / name
        if target.exists() and not force and _sha256(target) == digest:
            print(f"ok       {target}")
            continue
        print(f"download {url}")
        tmp = target.with_suffix(".part")
        req = urllib.request.Request(url, headers={"User-Agent": "bill-redact"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        got = _sha256(tmp)
        if got != digest:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {name}: expected {digest}, got {got}")
        tmp.replace(target)
        print(f"ok       {target}")
    return dest
