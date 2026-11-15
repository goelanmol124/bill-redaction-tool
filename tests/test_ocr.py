"""End-to-end test on a simulated scan. Needs the ONNX models:

    bill-redact fetch-models --dir models     (or set BILL_REDACT_MODELS)
"""
import os
from pathlib import Path

import pytest

MODELS = Path(os.environ.get("BILL_REDACT_MODELS", Path(__file__).resolve().parents[1] / "models"))
pytestmark = pytest.mark.skipif(not (MODELS / "det.onnx").exists(), reason="OCR models not downloaded")


def test_scanned_bundle(bundle, synthetic, example_config, tmp_path):
    pytest.importorskip("onnxruntime")
    from bill_redact import process_file
    from bill_redact.ocr.onnx_ocr import OnnxOcr

    scanned = synthetic.scan(bundle, tmp_path / "scanned.pdf")
    res = process_file(scanned, tmp_path / "out.pdf", example_config, engine="onnx", ocr=OnnxOcr(MODELS))
    assert res.ok
    for v in (10260.0, 2821.0, 13081.0, 1569.72, 14650.72, 85.5, 62.0, 120.0, 45.5):
        assert v in res.plan.values
