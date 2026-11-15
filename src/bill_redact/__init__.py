"""bill-redact: redact amounts, rates and quantities from bills and invoices.

Finds the sensitive numbers through the arithmetic that ties them together
(line amounts that add up to totals, amount / quantity = rate) instead of a
fixed layout, then paints over them on a raster copy of the page.
"""

from .analysis import RedactionPlan, TextBox, plan_redaction
from .config import AmountConfig, DocumentKind, QuantityConfig, RedactConfig, load_config
from .pipeline import process_file

__all__ = [
    "AmountConfig", "DocumentKind", "QuantityConfig", "RedactConfig", "RedactionPlan",
    "TextBox", "load_config", "plan_redaction", "process_file",
]
__version__ = "0.2.0"
