"""Generate a fictional document bundle for the demo and the tests.

Everything here is made up: the parties, items, numbers and layout. The
bundle has three pages: a cover sheet, an invoice and a delivery note that
repeat the same items, which is the situation the tool is built for (the
same commercial values appear on several documents of one shipment).

    python examples/make_synthetic.py            # writes examples/data/
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pymupdf as fitz
import numpy as np
from PIL import Image, ImageFilter

SELLER = "Example Supplies Co. (fictional)"
BUYER = "Demo Buyer Co. (fictional)"

ITEMS = [  # description, quantity (kg), rate, amount
    ("Copier paper, A4", "120.000", "85.50", "10,260.00"),
    ("Cardboard sheets", "45.500", "62.00", "2,821.00"),
]
SUBTOTAL, TAX, TOTAL = "13,081.00", "1,569.72", "14,650.72"
WORDS = "Fourteen Thousand Six Hundred Fifty Dollars and Seventy Two Cents Only"

FONT = "helv"


def _cell(page, x0, y0, x1, y1, text, size=10, bold=False, align=0):
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(0.3, 0.3, 0.3), width=0.6)
    if text:
        font = "hebo" if bold else FONT
        x = x0 + 4
        if align == 2:  # right-aligned
            x = x1 - 4 - fitz.get_text_length(text, fontname=font, fontsize=size)
        page.insert_text((x, y1 - 6), text, fontsize=size, fontname=font)


def _header(page, title, ref):
    page.insert_text((50, 60), title, fontsize=18, fontname="hebo")
    page.insert_text((50, 85), SELLER, fontsize=10, fontname=FONT)
    page.insert_text((50, 99), "1 Sample Street, Exampletown", fontsize=9, fontname=FONT)
    page.insert_text((360, 85), f"Ref: {ref}", fontsize=10, fontname=FONT)
    page.insert_text((360, 99), "Date: 2025-01-15", fontsize=10, fontname=FONT)
    page.insert_text((50, 125), f"Bill to / Ship to: {BUYER}", fontsize=10, fontname=FONT)


def build_bundle(path: Path) -> Path:
    doc = fitz.open()
    w, h = 595, 842  # A4 in points

    # page 0: cover sheet (no keywords, no amounts)
    p = doc.new_page(width=w, height=h)
    p.insert_text((50, 80), "Shipment documents", fontsize=20, fontname="hebo")
    p.insert_text((50, 110), "Attached: commercial paperwork for one consignment.", fontsize=11, fontname=FONT)
    p.insert_text((50, 130), "Prepared for internal filing. Pages: 3", fontsize=11, fontname=FONT)

    # page 1: invoice
    p = doc.new_page(width=w, height=h)
    _header(p, "INVOICE", "DEMO-0001")
    cols = [50, 80, 270, 370, 450, 545]
    y = 150
    for i, t in enumerate(["#", "Description", "Quantity", "Rate", "Amount"]):
        _cell(p, cols[i], y, cols[i + 1], y + 20, t, bold=True)
    for n, (desc, qty, rate, amount) in enumerate(ITEMS, start=1):
        y += 20
        for i, t in enumerate([str(n), desc, f"{qty} KG", rate, amount]):
            _cell(p, cols[i], y, cols[i + 1], y + 20, t, align=2 if i >= 2 else 0)
    for label, value in [("Subtotal", SUBTOTAL), ("Tax @ 12%", TAX), ("Total", TOTAL)]:
        y += 20
        _cell(p, cols[3], y, cols[4], y + 20, label, bold=True)
        _cell(p, cols[4], y, cols[5], y + 20, value, align=2)
    y += 35
    p.insert_text((50, y), f"Amount in words: {WORDS}", fontsize=10, fontname=FONT)
    p.insert_text((50, y + 40), "Terms: payment within 30 days.", fontsize=9, fontname=FONT)

    # page 2: delivery note for the same consignment
    p = doc.new_page(width=w, height=h)
    _header(p, "DELIVERY NOTE", "DEMO-0001-D")
    cols = [50, 80, 330, 450, 545]
    y = 150
    for i, t in enumerate(["#", "Description", "Net weight", "Value"]):
        _cell(p, cols[i], y, cols[i + 1], y + 20, t, bold=True)
    for n, (desc, qty, _rate, amount) in enumerate(ITEMS, start=1):
        y += 20
        for i, t in enumerate([str(n), desc, f"{qty} KG", amount]):
            _cell(p, cols[i], y, cols[i + 1], y + 20, t, align=2 if i >= 2 else 0)
    y += 20
    _cell(p, cols[1], y, cols[2], y + 20, "Total", bold=True)
    _cell(p, cols[2], y, cols[3], y + 20, "165.500 KG", align=2)
    _cell(p, cols[3], y, cols[4], y + 20, SUBTOTAL, align=2)
    y += 35
    p.insert_text((50, y), "Packages: 6     Vehicle: DEMO-TRUCK-7", fontsize=10, fontname=FONT)
    p.insert_text((50, y + 20), "Received in good condition. Signature: ______________", fontsize=10,
                  fontname=FONT)

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    return path


def scan(src: Path, dst: Path, dpi: int = 150, seed: int = 7) -> Path:
    """Simulate a scanner: rasterise, tilt slightly, blur, add noise, JPEG."""
    rng = np.random.default_rng(seed)
    out = fitz.open()
    with fitz.open(src) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = img.rotate(float(rng.uniform(-1.2, 1.2)), expand=False, fillcolor=(255, 255, 255),
                             resample=Image.BICUBIC)
            img = img.filter(ImageFilter.GaussianBlur(0.6))
            arr = np.asarray(img).astype(np.int16) + rng.normal(0, 6, (img.height, img.width, 1)).astype(np.int16)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("L")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            np_page = out.new_page(width=page.rect.width, height=page.rect.height)
            np_page.insert_image(np_page.rect, stream=buf.getvalue())
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, garbage=4, deflate=True)
    out.close()
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    args = ap.parse_args()
    out = Path(args.out)
    digital = build_bundle(out / "synthetic_bundle.pdf")
    scanned = scan(digital, out / "synthetic_bundle_scanned.pdf")
    print(f"wrote {digital}\nwrote {scanned}")


if __name__ == "__main__":
    main()
