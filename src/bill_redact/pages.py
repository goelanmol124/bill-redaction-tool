"""Page input/output: PDFs and images in, redacted raster PDF/PNG out.

Redacted pages are written as images. The original page (including any
hidden text layer) is replaced, so the covered values cannot be copied out of
the output file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import pymupdf as fitz
import numpy as np
from PIL import Image

from .analysis import TextBox

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Page:
    image: Image.Image                 # RGB raster the boxes refer to
    text_boxes: Optional[List[TextBox]] = None  # from the PDF text layer, if any
    pdf_rect: Optional[Tuple[float, float]] = None  # page size in points
    meta: dict = field(default_factory=dict)


def _text_layer_boxes(page: "fitz.Page", scale: float) -> List[TextBox]:
    """One TextBox per text line of a born-digital PDF page, in pixel units."""
    boxes: List[TextBox] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            boxes.append(TextBox.from_rect(text, x0 * scale, y0 * scale, x1 * scale, y1 * scale))
    return boxes


def load_pages(path: str | Path, dpi: int = 200, use_text_layer: bool = True,
               min_text_chars: int = 20) -> List[Page]:
    """Load a PDF or image file as a list of :class:`Page`.

    For PDF pages with a usable text layer (``min_text_chars`` or more) the
    text boxes come from the PDF itself and OCR can be skipped. Scanned pages
    with a single embedded image use that image at full resolution; other
    pages are rendered at ``dpi``.
    """
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXTS:
        return [Page(Image.open(path).convert("RGB"))]

    pages: List[Page] = []
    with fitz.open(path) as doc:
        for page in doc:
            rect = (page.rect.width, page.rect.height)
            has_text = use_text_layer and len(page.get_text("text").strip()) >= min_text_chars
            imgs = page.get_images(full=True)
            if not has_text and len(imgs) == 1:
                raw = doc.extract_image(imgs[0][0])["image"]
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                pages.append(Page(img, None, rect, {"source": "embedded-image"}))
                continue
            scale = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            boxes = _text_layer_boxes(page, pix.width / page.rect.width) if has_text else None
            pages.append(Page(img, boxes, rect, {"source": "text-layer" if has_text else "render"}))
    return pages


def paint_boxes(image: Image.Image, boxes: Sequence[TextBox], fill=(0, 0, 0), pad: int = 2) -> Image.Image:
    """Return a copy of ``image`` with each box covered by a solid rectangle."""
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    color = tuple(int(c) for c in fill)
    for b in boxes:
        x0, y0, x1, y1 = b.bbox
        p0 = (max(0, int(x0) - pad), max(0, int(y0) - pad))
        p1 = (min(w - 1, int(round(x1)) + pad), min(h - 1, int(round(y1)) + pad))
        cv2.rectangle(arr, p0, p1, color, thickness=-1)
    return Image.fromarray(arr)


def write_pdf(src: str | Path, pages: Sequence[Page], replaced: dict, out: str | Path) -> None:
    """Write ``out``: pages listed in ``replaced`` (index -> image) become
    image-only pages; the others are copied from ``src`` (or rasterised when
    the source is an image file). Document metadata is cleared."""
    src = Path(src)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    if src.suffix.lower() in IMAGE_EXTS:
        doc = fitz.open()
        for i, p in enumerate(pages):
            img = replaced.get(i, p.image)
            pg = doc.new_page(width=img.width * 72 / 150, height=img.height * 72 / 150)
            pg.insert_image(pg.rect, stream=png_bytes(img))
    else:
        doc = fitz.open(src)
        for i, img in sorted(replaced.items()):
            rect = doc[i].rect
            doc.delete_page(i)
            pg = doc.new_page(pno=i, width=rect.width, height=rect.height)
            pg.insert_image(pg.rect, stream=png_bytes(img))
    doc.set_metadata({})
    try:
        doc.del_xml_metadata()
    except Exception:  # pragma: no cover - older PyMuPDF
        pass
    doc.save(out, garbage=4, deflate=True)
    doc.close()
