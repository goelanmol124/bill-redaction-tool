"""Decide *what* to redact from OCR output, using arithmetic, not layout.

The idea: on commercial documents the sensitive numbers are tied together by
arithmetic. Line amounts add up to subtotals, taxes and totals; an amount
divided by a quantity gives a unit rate that is also printed somewhere. So
instead of knowing where a vendor puts its "Total" box, we look for numbers
that take part in those relations:

1. Quantities are numbers followed by a unit (``12.500 KG``), numbers with a
   unit in the box to their right, or numbers with a "quantity-like" number of
   decimal places (configurable).
2. Every other number on the document pages is a candidate amount.
3. Subset-sum search: any group of 2..N candidates whose sum equals another
   candidate (within a tolerance) is redacted, together with that sum.
4. Rate inference: for every redacted amount and every quantity,
   ``amount / quantity`` is computed; a candidate equal to it is a unit rate
   and is redacted too.
5. Quantities themselves, and boxes that spell an amount in words
   ("... Only"), are redacted.

Nothing here depends on a particular document layout. Page types, units,
tolerances and the words-pattern come from :class:`~bill_redact.config.RedactConfig`.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import RedactConfig

log = logging.getLogger(__name__)

# A number token: 1234 / 1,234 / 1,23,456 (grouped) with optional decimals.
# Rejects tokens glued to letters (IDs such as AB12CD), parts of dates and
# codes joined by "-" or "/" (2025-01-15, 12/03/24, X-7) and percentages.
NUMBER_RE = re.compile(r"(?<![\w.,/-])(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?(?![\w%]|[.,/-]\d)")


def _key(value: float) -> float:
    """Normalise a float for set membership (avoid 0.1+0.2 noise)."""
    return round(float(value), 4)


@dataclass
class TextBox:
    """One OCR result: recognised text and its quadrilateral in pixels."""

    text: str
    quad: List[Tuple[float, float]]  # 4 points, clockwise from top-left
    confidence: float = 1.0

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    @property
    def height(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return y1 - y0

    @classmethod
    def from_rect(cls, text: str, x0: float, y0: float, x1: float, y1: float, confidence: float = 1.0) -> "TextBox":
        return cls(text, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], confidence)


@dataclass
class ParsedNumber:
    value: float
    decimals: int
    grouped: bool  # had thousands separators
    start: int
    end: int


def parse_numbers(text: str) -> List[ParsedNumber]:
    out: List[ParsedNumber] = []
    for m in NUMBER_RE.finditer(text):
        int_part, frac = m.group(1), m.group(2)
        try:
            value = float(int_part.replace(",", "") + ("." + frac if frac else ""))
        except ValueError:
            continue
        out.append(ParsedNumber(value, len(frac) if frac else 0, "," in int_part, m.start(), m.end()))
    return out


# --------------------------------------------------------------------------
# page classification
# --------------------------------------------------------------------------

def classify_pages(pages: Sequence[Sequence[TextBox]], cfg: RedactConfig) -> Dict[str, int]:
    """Map each configured document kind to the first page that matches it.

    Without configured kinds every non-skipped page is its own "document",
    named ``page<N>``.
    """
    skip = set(cfg.skip_pages)
    if not cfg.documents:
        return {f"page{i}": i for i in range(len(pages)) if i not in skip}

    found: Dict[str, int] = {}
    for idx, boxes in enumerate(pages):
        if idx in skip:
            continue
        joined = " ".join(b.text for b in boxes).lower()
        scores = []
        for kind in cfg.documents:
            hits = sum(joined.count(k.lower()) for k in kind.keywords)
            scores.append((hits, kind.name))
        best_hits = max(s[0] for s in scores)
        if best_hits == 0:
            continue
        # Highest keyword count wins; ties go to the kind listed first.
        best = next(name for hits, name in scores if hits == best_hits)
        if best not in found:
            found[best] = idx
            log.info("page %d classified as %s", idx, best)
    return found


# --------------------------------------------------------------------------
# quantities
# --------------------------------------------------------------------------

def _unit_regex(units: Iterable[str]) -> re.Pattern:
    alts = "|".join(sorted((re.escape(u) for u in units), key=len, reverse=True))
    return re.compile(rf"^[\s|/:]*({alts})(?![A-Za-z])", re.IGNORECASE)


def extract_quantities(boxes: Sequence[TextBox], cfg: RedactConfig, page_width: Optional[float] = None) -> List[float]:
    q = cfg.quantity
    unit_after = _unit_regex(q.units)
    units_upper = {u.upper() for u in q.units}
    found: List[float] = []

    def in_range(v: float) -> bool:
        return q.min_value <= v <= q.max_value

    # 1) number and unit in the same box
    for b in boxes:
        for n in parse_numbers(b.text):
            followed_by_unit = bool(unit_after.match(b.text[n.end:]))
            value = n.value
            if followed_by_unit and n.grouped and n.decimals == 0 and q.grouped_integer_scale:
                value = value / q.grouped_integer_scale
            if followed_by_unit and n.decimals in q.decimal_places_with_unit and in_range(value):
                found.append(value)
            elif n.decimals > 0 and n.decimals in q.decimal_places_without_unit and in_range(value):
                found.append(value)

    # 2) bare number with a unit-only box just to its right on the same line
    if page_width is None:
        page_width = max((b.bbox[2] for b in boxes), default=1.0)
    max_gap = q.unit_max_gap_frac * page_width
    numbers = []
    for b in boxes:
        parsed = parse_numbers(b.text.strip())
        if len(parsed) == 1 and parsed[0].start == 0 and parsed[0].end == len(b.text.strip()):
            numbers.append((b, parsed[0]))
    unit_boxes = [b for b in boxes if b.text.strip().strip("|/:").strip().upper() in units_upper]
    for b, n in numbers:
        if n.decimals not in q.decimal_places_with_unit or not in_range(n.value):
            continue
        cx, cy = b.center
        for u in unit_boxes:
            ux, uy = u.center
            same_line = abs(uy - cy) < 0.6 * max(b.height, u.height, 1.0)
            if same_line and 0 < ux - cx and u.bbox[0] - b.bbox[2] < max_gap:
                found.append(n.value)
                break

    uniq = sorted({_key(v) for v in found if v != 0})
    log.info("quantities: %s", uniq)
    return uniq


# --------------------------------------------------------------------------
# amounts
# --------------------------------------------------------------------------

def candidate_numbers(boxes: Sequence[TextBox], cfg: RedactConfig) -> List[float]:
    out: List[float] = []
    for b in boxes:
        for n in parse_numbers(b.text):
            if n.decimals == 0 and n.value <= cfg.amounts.ignore_integers_up_to:
                continue
            out.append(n.value)
    return out


def _close(a: float, b: float, rel: float) -> bool:
    if b == 0:
        return a == 0
    return abs(a - b) / abs(b) <= rel


def split_amounts(numbers: Sequence[float], quantities: Sequence[float], rel: float) -> Tuple[List[float], List[float]]:
    """Numbers within ``rel`` of a known quantity are quantities; the rest are amounts."""
    amounts, qtys = [], list(quantities)
    for v in numbers:
        if any(_close(v, q, rel) for q in quantities if q != 0):
            qtys.append(v)
        else:
            amounts.append(v)
    return amounts, qtys


def find_sum_relations(values: Sequence[float], cfg: RedactConfig) -> Set[float]:
    """Values that are the sum of 2..max_terms other values, plus those addends."""
    a = cfg.amounts
    counts = Counter(_key(v) for v in values if v > 0)
    pool = sorted(itertools.chain.from_iterable([v] * min(c, a.max_repeats) for v, c in counts.items()))
    targets = sorted(counts)
    n = len(pool)
    hits: Set[float] = set()
    for size in range(2, min(a.max_terms, n) + 1):
        if math.comb(n, size) > a.combination_budget:
            log.warning("subset search stopped at %d terms (%d candidates exceed the budget)", size - 1, n)
            break
        for combo in itertools.combinations(pool, size):
            s = sum(combo)
            i = bisect.bisect_left(targets, s - a.sum_tolerance)
            while i < len(targets) and targets[i] <= s + a.sum_tolerance:
                t = targets[i]
                if t > combo[-1]:  # the total is larger than every addend
                    hits.update(combo)
                    hits.add(t)
                    break
                i += 1
    return hits


def infer_rates(redacted: Set[float], quantities: Sequence[float], amounts: Iterable[float], rel: float) -> Set[float]:
    """Amounts equal to (redacted amount / quantity) are unit rates."""
    amounts = set(_key(v) for v in amounts)
    rates: Set[float] = set()
    for amount in redacted:
        for q in set(quantities):
            if q <= 0:
                continue
            rate = amount / q
            for tv in amounts:
                if rate > 0 and abs(rate - tv) / rate < rel:
                    rates.add(tv)
                    break
    return rates


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

@dataclass
class RedactionPlan:
    document_pages: Dict[str, int]
    quantities: List[float]
    amounts: Dict[str, List[float]]
    values: Set[float]
    boxes: Dict[int, List[TextBox]] = field(default_factory=dict)
    ok: bool = True
    reason: str = ""

    def summary(self, include_values: bool = False) -> dict:
        """Report for logs. The redacted values themselves are left out unless
        asked for, so the report does not leak what was redacted."""
        d = {
            "ok": self.ok,
            "reason": self.reason,
            "document_pages": self.document_pages,
            "n_quantities": len(self.quantities),
            "n_values": len(self.values),
            "boxes_per_page": {p: len(b) for p, b in sorted(self.boxes.items())},
        }
        if include_values:
            d["quantities"] = self.quantities
            d["values"] = sorted(self.values)
        return d


def _box_matches(box: TextBox, values: Set[float], cfg: RedactConfig, word_res, extra_res) -> bool:
    text = box.text
    for n in parse_numbers(text):
        if _key(n.value) in values:
            return True
        if n.grouped and n.decimals == 0 and cfg.quantity.grouped_integer_scale:
            if _key(n.value / cfg.quantity.grouped_integer_scale) in values:
                return True
    if any(r.search(text) for r in word_res):
        return True
    if any(r.search(text) for r in extra_res):
        return True
    return False


def plan_redaction(pages: Sequence[Sequence[TextBox]], cfg: RedactConfig,
                   page_widths: Optional[Sequence[float]] = None) -> RedactionPlan:
    """Work out which values and boxes to redact across a bundle of pages."""
    doc_pages = classify_pages(pages, cfg)
    if cfg.documents and cfg.fail_closed:
        missing = [d.name for d in cfg.documents if d.name not in doc_pages]
        if missing:
            return RedactionPlan(doc_pages, [], {}, set(), ok=False,
                                 reason=f"document kind(s) not found: {', '.join(missing)}")

    q_pages = range(len(pages)) if cfg.quantity_pages == "all" else sorted(doc_pages.values())
    quantities: List[float] = []
    for i in q_pages:
        width = page_widths[i] if page_widths else None
        quantities.extend(extract_quantities(pages[i], cfg, width))
    quantities = sorted(set(quantities))

    rel = cfg.amounts.relative_tolerance
    amounts: Dict[str, List[float]] = {}
    for name, idx in doc_pages.items():
        nums = candidate_numbers(pages[idx], cfg)
        doc_amounts, more_q = split_amounts(nums, quantities, rel)
        amounts[name] = doc_amounts
        quantities = sorted(set(_key(v) for v in more_q))

    pooled = [v for vals in amounts.values() for v in vals]
    values = find_sum_relations(pooled, cfg)
    values |= infer_rates(values, quantities, pooled, rel)
    if cfg.quantity.redact:
        values |= {_key(q) for q in quantities if q != 0}

    plan = RedactionPlan(doc_pages, quantities, amounts, values)

    if cfg.fail_closed:
        if cfg.documents:
            empty = [name for name, vals in amounts.items()
                     if not any(_key(v) in values for v in vals)]
        else:  # pages are pooled: require at least one relation somewhere
            empty = [] if any(_key(v) in values for v in pooled) else ["all pages"]
        if empty:
            plan.ok = False
            plan.reason = f"no related amounts found on: {', '.join(empty)}"
            return plan

    word_res = [re.compile(p, re.IGNORECASE) for p in cfg.amount_in_words_patterns]
    extra_res = [re.compile(p, re.IGNORECASE) for p in cfg.extra_patterns]
    target_pages = range(len(pages)) if cfg.redact_all_pages else sorted(doc_pages.values())
    for i in target_pages:
        hits = [b for b in pages[i] if _box_matches(b, values, cfg, word_res, extra_res)]
        if hits:
            plan.boxes[i] = hits
    return plan
