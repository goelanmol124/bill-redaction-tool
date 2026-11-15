"""Configuration for the redaction heuristics.

Everything that depends on *which* documents you process lives here, not in the
code: which page types to look for, which units mark a quantity, how amounts
written in words are recognised, tolerances, and the fill colour. Load a YAML
file with :func:`load_config`, or build :class:`RedactConfig` directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


@dataclass
class DocumentKind:
    """A page type that must be present in a bundle, found by keywords.

    A page is assigned to the kind whose keywords occur most often in its OCR
    text (case-insensitive substring match).
    """

    name: str
    keywords: List[str]


@dataclass
class QuantityConfig:
    """How quantities (weights, counts, volumes) are recognised.

    Quantities are redacted too (unless ``redact`` is false) and they are used to
    infer unit rates: ``rate = amount / quantity``.
    """

    units: List[str] = field(default_factory=lambda: [
        "KG", "KGS", "G", "T", "MT", "MTS", "TON", "TONS", "TONNES",
        "L", "LTR", "LTRS", "PCS", "NOS", "UNITS", "BOX", "BOXES",
    ])
    # A decimal with one of these numbers of decimal places counts as a quantity
    # when a unit is in the same text box or just to its right.
    decimal_places_with_unit: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # A decimal with one of these numbers of decimal places counts as a quantity
    # even without a unit (many forms print weights with 3 decimals and money
    # with 2). Set to [] to disable.
    decimal_places_without_unit: List[int] = field(default_factory=lambda: [3])
    min_value: float = 0.001
    max_value: float = 1_000_000.0
    # Unit in a separate OCR box: it must sit to the right of the number, on the
    # same line, at most this fraction of the page width away.
    unit_max_gap_frac: float = 0.15
    # Optional: divide grouped integers next to a unit by this factor, e.g. 1000
    # to read "12,500 KG" as 12.5 when the other document states tonnes.
    # null/None disables it (default).
    grouped_integer_scale: Optional[float] = None
    redact: bool = True


@dataclass
class AmountConfig:
    """Parameters for finding monetary values via arithmetic relations."""

    # Integers up to this value are ignored (serial numbers, page numbers,
    # line numbers ...).
    ignore_integers_up_to: int = 5
    # Two values "match" when they differ by at most this fraction.
    relative_tolerance: float = 0.01
    # A subset "sums to" a target when the difference is at most this.
    sum_tolerance: float = 1.0
    # Largest number of addends tried in the subset-sum search.
    max_terms: int = 6
    # Upper bound on subset combinations evaluated per subset size. Larger
    # sizes are skipped once a size would exceed this budget.
    combination_budget: int = 2_000_000
    # Each distinct value is used at most this many times as an addend.
    max_repeats: int = 2


@dataclass
class RedactConfig:
    documents: List[DocumentKind] = field(default_factory=list)
    # Page indices (0-based) never used for classification. They still donate
    # quantities and, with ``redact_all_pages``, get redacted.
    skip_pages: List[int] = field(default_factory=list)
    # Pages used for quantity extraction: "all" or "documents".
    quantity_pages: str = "all"
    # Redact every page with the final value set, not only the document pages.
    redact_all_pages: bool = False
    quantity: QuantityConfig = field(default_factory=QuantityConfig)
    amounts: AmountConfig = field(default_factory=AmountConfig)
    # Regexes (Python syntax, case-insensitive) for amounts written in words.
    # An OCR box whose text matches is redacted as a whole.
    amount_in_words_patterns: List[str] = field(default_factory=lambda: [
        r"\b(rupees?|inr|rs\.?|dollars?|usd|euros?|eur|pounds?|gbp)\b.*\bonly\b",
    ])
    # Extra regexes: any OCR box matching one of them is always redacted.
    extra_patterns: List[str] = field(default_factory=list)
    # Fill colour (RGB) and padding in pixels for redaction boxes.
    fill: Sequence[int] = (0, 0, 0)
    pad: int = 2
    # Refuse to write output unless every document page has at least one
    # redacted monetary value (the original tool's validation step).
    fail_closed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build(cls, data: Dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} option(s): {', '.join(sorted(unknown))}")
    kwargs = {}
    for key, value in data.items():
        if key == "documents":
            kwargs[key] = [DocumentKind(name=d["name"], keywords=list(d["keywords"])) for d in value or []]
        elif key == "quantity":
            kwargs[key] = _build(QuantityConfig, value or {})
        elif key == "amounts":
            kwargs[key] = _build(AmountConfig, value or {})
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_dict(data: Optional[Dict[str, Any]]) -> RedactConfig:
    cfg = _build(RedactConfig, data or {})
    if cfg.quantity_pages not in ("all", "documents"):
        raise ValueError("quantity_pages must be 'all' or 'documents'")
    if len(cfg.fill) != 3:
        raise ValueError("fill must be an RGB triple")
    return cfg


def load_config(path: Optional[str | Path]) -> RedactConfig:
    """Load a YAML config; ``None`` returns the defaults."""
    if path is None:
        return RedactConfig()
    with open(path, "r", encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))


def dump_default_config() -> str:
    def plain(obj):
        if is_dataclass(obj):
            return {k: plain(v) for k, v in asdict(obj).items()}
        if isinstance(obj, tuple):
            return list(obj)
        return obj
    return yaml.safe_dump(plain(RedactConfig()), sort_keys=False)
