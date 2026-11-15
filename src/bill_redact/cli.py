"""Command line interface: ``bill-redact``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from .config import dump_default_config, load_config
from .pages import IMAGE_EXTS

INPUT_EXTS = {".pdf"} | IMAGE_EXTS


def _inputs(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in INPUT_EXTS))
        else:
            files.append(p)
    return files


def _make_ocr(args):
    if args.engine == "text":
        return None
    from .ocr.onnx_ocr import OnnxOcr

    class Lazy:
        """Load the models only when a page actually needs OCR."""
        engine = None

        def __call__(self, image):
            if Lazy.engine is None:
                Lazy.engine = OnnxOcr(args.models_dir)
            return Lazy.engine(image)

    return Lazy()


def cmd_redact(args) -> int:
    from .pipeline import process_file

    cfg = load_config(args.config)
    ocr = _make_ocr(args)
    files = _inputs(args.inputs)
    if not files:
        print("no input files found", file=sys.stderr)
        return 2
    out_dir = Path(args.output)
    failures = 0
    for f in files:
        suffix = ".pdf" if f.suffix.lower() == ".pdf" else ".png"
        target = out_dir / (f.stem + suffix)
        res = process_file(f, target, cfg, engine=args.engine, ocr=ocr, dpi=args.dpi, report=not args.no_report)
        status = "REDACTED" if res.ok else "SKIPPED"
        print(f"{status:9s} {f.name}  ({res.seconds:.1f}s){'' if res.ok else '  ' + res.plan.reason}")
        failures += 0 if res.ok else 1
    return 0 if failures == 0 else 1


def cmd_ocr(args) -> int:
    from .pages import load_pages
    from .pipeline import ocr_pages

    pages = load_pages(args.input, dpi=args.dpi, use_text_layer=args.engine != "onnx")
    boxes = ocr_pages(pages, args.engine, _make_ocr(args))
    data = [[{"text": b.text, "bbox": [round(v, 1) for v in b.bbox], "confidence": round(b.confidence, 3)}
             for b in page] for page in boxes]
    print(json.dumps(data, indent=1, ensure_ascii=False))
    return 0


def cmd_plan(args) -> int:
    """Dry run: show what would be redacted, including the values."""
    from .analysis import plan_redaction
    from .pages import load_pages
    from .pipeline import ocr_pages

    cfg = load_config(args.config)
    pages = load_pages(args.input, dpi=args.dpi, use_text_layer=args.engine != "onnx")
    boxes = ocr_pages(pages, args.engine, _make_ocr(args))
    plan = plan_redaction(boxes, cfg, page_widths=[p.image.width for p in pages])
    summary = plan.summary(include_values=True)
    summary["boxes"] = {p: [b.text for b in bs] for p, bs in plan.boxes.items()}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if plan.ok else 1


def cmd_fetch(args) -> int:
    from .models import fetch_models
    fetch_models(args.dir, args.variant, args.force)
    return 0


def cmd_config(_args) -> int:
    print(dump_default_config(), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bill-redact",
                                description="Redact amounts, rates and quantities from bills and invoices.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, config=True):
        sp.add_argument("--engine", choices=["auto", "text", "onnx"], default="auto",
                        help="auto: PDF text layer when present, else OCR (default)")
        sp.add_argument("--models-dir", default=None, help="folder with det.onnx/rec.onnx[/cls.onnx]")
        sp.add_argument("--dpi", type=int, default=200, help="render resolution for PDF pages")
        if config:
            sp.add_argument("-c", "--config", default=None, help="YAML config (see `bill-redact config`)")

    r = sub.add_parser("redact", help="redact PDFs/images or whole folders")
    r.add_argument("inputs", nargs="+")
    r.add_argument("-o", "--output", default="output", help="output folder (default: output/)")
    r.add_argument("--no-report", action="store_true", help="do not write the .json report")
    common(r)
    r.set_defaults(func=cmd_redact)

    pl = sub.add_parser("plan", help="dry run: print what would be redacted")
    pl.add_argument("input")
    common(pl)
    pl.set_defaults(func=cmd_plan)

    o = sub.add_parser("ocr", help="print text boxes as JSON")
    o.add_argument("input")
    common(o, config=False)
    o.set_defaults(func=cmd_ocr)

    f = sub.add_parser("fetch-models", help="download the public PP-OCR ONNX models")
    f.add_argument("--dir", default="models")
    f.add_argument("--variant", choices=["mobile", "server"], default="mobile")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("config", help="print the default configuration as YAML")
    c.set_defaults(func=cmd_config)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
