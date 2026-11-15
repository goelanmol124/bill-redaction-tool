import json
from pathlib import Path

import pymupdf as fitz

from bill_redact import process_file
from bill_redact.cli import main
from bill_redact.config import config_from_dict

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config.yaml"
SECRETS = ["10,260.00", "2,821.00", "13,081.00", "1,569.72", "14,650.72", "85.50", "62.00"]


def test_born_digital_bundle_is_redacted(bundle, example_config, tmp_path):
    out = tmp_path / "out.pdf"
    res = process_file(bundle, out, example_config, engine="text")
    assert res.ok
    assert res.plan.document_pages == {"invoice": 1, "delivery_note": 2}
    assert res.plan.boxes[1] and res.plan.boxes[2]
    with fitz.open(out) as doc:
        assert doc.page_count == 3
        assert "Shipment documents" in doc[0].get_text()   # untouched page keeps its text
        for i in (1, 2):
            assert doc[i].get_text().strip() == ""          # redacted pages are raster only
        assert not any(doc.metadata.get(k) for k in ("author", "creator", "producer", "title"))
    report = json.loads((tmp_path / "out.pdf.json").read_text())
    assert report["ok"] and "values" not in report


def test_redacted_values_are_covered(bundle, example_config, tmp_path):
    res = process_file(bundle, tmp_path / "out.pdf", example_config, engine="text")
    covered = {b.text for page in res.plan.boxes.values() for b in page}
    for s in SECRETS:
        assert any(s in t for t in covered), s
    assert not any("DEMO-0001" in t for t in covered)


def test_missing_document_writes_nothing(bundle, tmp_path):
    cfg = config_from_dict({"documents": [{"name": "receipt", "keywords": ["receipt of payment"]}]})
    out = tmp_path / "none.pdf"
    res = process_file(bundle, out, cfg, engine="text")
    assert not res.ok and not out.exists()


def test_cli_redact_and_config(bundle, tmp_path, capsys):
    rc = main(["redact", str(bundle), "-c", str(CONFIG), "--engine", "text", "-o", str(tmp_path / "o")])
    assert rc == 0 and (tmp_path / "o" / "bundle.pdf").exists()
    assert main(["config"]) == 0
    assert "amount_in_words_patterns" in capsys.readouterr().out
