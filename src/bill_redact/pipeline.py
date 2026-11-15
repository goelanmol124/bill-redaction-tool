"""End-to-end processing of one document bundle (PDF or image)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .analysis import RedactionPlan, TextBox, plan_redaction
from .config import RedactConfig
from .pages import Page, load_pages, paint_boxes, write_pdf

log = logging.getLogger(__name__)

OcrFn = Callable[[object], List[TextBox]]


@dataclass
class Result:
    input: str
    output: Optional[str]
    plan: RedactionPlan
    seconds: float

    @property
    def ok(self) -> bool:
        return self.plan.ok and self.output is not None

    def to_dict(self) -> dict:
        d = {"input": self.input, "output": self.output, "seconds": round(self.seconds, 2)}
        d.update(self.plan.summary())
        return d


def ocr_pages(pages: List[Page], engine: str = "auto", ocr: Optional[OcrFn] = None) -> List[List[TextBox]]:
    """Text boxes for each page.

    ``engine``: ``text`` uses only the PDF text layer, ``onnx`` always runs OCR,
    ``auto`` uses the text layer when a page has one and OCR otherwise.
    """
    out: List[List[TextBox]] = []
    for i, p in enumerate(pages):
        if engine in ("auto", "text") and p.text_boxes is not None:
            out.append(p.text_boxes)
            continue
        if engine == "text":
            log.warning("page %d has no text layer; skipped (use --engine auto or onnx)", i)
            out.append([])
            continue
        if ocr is None:
            raise RuntimeError("page %d needs OCR but no OCR engine was given" % i)
        out.append(ocr(p.image))
    return out


def process_file(path: str | Path, out_path: str | Path, cfg: RedactConfig, *, engine: str = "auto",
                 ocr: Optional[OcrFn] = None, dpi: int = 200, report: bool = True) -> Result:
    t0 = time.perf_counter()
    path = Path(path)
    pages = load_pages(path, dpi=dpi, use_text_layer=engine != "onnx")
    boxes = ocr_pages(pages, engine, ocr)
    plan = plan_redaction(boxes, cfg, page_widths=[p.image.width for p in pages])

    output = None
    if plan.ok:
        replaced = {i: paint_boxes(pages[i].image, b, cfg.fill, cfg.pad) for i, b in plan.boxes.items()}
        # Pages that were classified but needed no box still lose their text layer.
        for i in plan.document_pages.values():
            replaced.setdefault(i, pages[i].image)
        out_path = Path(out_path)
        if out_path.suffix.lower() == ".png" and len(pages) == 1:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            replaced.get(0, pages[0].image).save(out_path)
        else:
            write_pdf(path, pages, replaced, out_path)
        output = str(out_path)
        log.info("wrote %s (%d boxes)", out_path, sum(len(b) for b in plan.boxes.values()))
    else:
        log.warning("%s: not redacted (%s); no output written", path.name, plan.reason)

    result = Result(str(path), output, plan, time.perf_counter() - t0)
    if report and output:
        Path(str(out_path) + ".json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
