import pytest

from bill_redact.analysis import (
    TextBox, classify_pages, extract_quantities, find_sum_relations, parse_numbers, plan_redaction,
)
from bill_redact.config import DocumentKind, RedactConfig, config_from_dict


def tb(text, x0, y0, x1=None, y1=None):
    return TextBox.from_rect(text, x0, y0, x1 if x1 is not None else x0 + 10 * len(text), y1 or y0 + 20)


@pytest.mark.parametrize("text,expected", [
    ("10,260.00", [10260.0]),
    ("1,23,456.50", [123456.5]),          # grouping in twos is accepted too
    ("120.000 KG", [120.0]),
    ("Date: 2025-01-15", []),             # date parts are not numbers
    ("Ref AB12CD", []),                    # glued to letters
    ("Tax @ 12%", []),                     # percentages are skipped
    ("Qty 3 x 4.50", [3.0, 4.5]),
])
def test_parse_numbers(text, expected):
    assert [n.value for n in parse_numbers(text)] == expected


def test_sum_relations_find_totals_and_addends():
    hits = find_sum_relations([100.0, 250.0, 350.0, 42.0, 7.5], RedactConfig())
    assert {100.0, 250.0, 350.0} <= hits
    assert 42.0 not in hits and 7.5 not in hits


def test_sum_relations_respect_budget():
    cfg = config_from_dict({"amounts": {"combination_budget": 10}})
    # 10 values: C(10, 2) = 45 already exceeds the budget, so nothing is searched
    assert find_sum_relations([float(v) for v in range(10, 20)], cfg) == set()


def test_quantities_with_unit_inline_and_in_separate_box():
    boxes = [tb("12.500 KG", 0, 0), tb("40.00", 300, 100, 350, 120), tb("PCS", 360, 100, 390, 120),
             tb("99.95", 300, 200, 350, 220)]
    q = extract_quantities(boxes, RedactConfig(), page_width=1000)
    assert 12.5 in q and 40.0 in q
    assert 99.95 not in q


def test_grouped_integer_scale_is_opt_in():
    boxes = [tb("12,500 KG", 0, 0)]
    assert extract_quantities(boxes, RedactConfig(), 1000) == [12500.0]
    cfg = config_from_dict({"quantity": {"grouped_integer_scale": 1000}})
    assert extract_quantities(boxes, cfg, 1000) == [12.5]


def test_classification_uses_keyword_counts():
    cfg = RedactConfig(documents=[DocumentKind("a", ["alpha"]), DocumentKind("b", ["beta"])])
    pages = [[tb("alpha beta", 0, 0)], [tb("beta beta alpha", 0, 0)], [tb("alpha", 0, 0)]]
    assert classify_pages(pages, cfg) == {"a": 0, "b": 1}
    cfg.skip_pages = [0]
    assert classify_pages(pages, cfg) == {"b": 1, "a": 2}


def _bill():
    return [
        tb("Widget", 0, 0), tb("2.000 KG", 200, 0), tb("15.00", 400, 0), tb("30.00", 500, 0),
        tb("Gadget", 0, 30), tb("4.000 KG", 200, 30), tb("5.00", 400, 30), tb("20.00", 500, 30),
        tb("Total", 0, 60), tb("50.00", 500, 60),
        tb("Fifty dollars only", 0, 90), tb("Phone 4471", 0, 120),
    ]


def test_plan_redacts_amounts_rates_quantities_and_words():
    plan = plan_redaction([_bill()], RedactConfig())
    assert plan.ok
    texts = {b.text for b in plan.boxes[0]}
    assert {"30.00", "20.00", "50.00", "15.00", "5.00", "2.000 KG", "4.000 KG", "Fifty dollars only"} <= texts
    assert "Phone 4471" not in texts and "Widget" not in texts


def test_plan_fails_closed_without_relations():
    plan = plan_redaction([[tb("Hello", 0, 0), tb("123.45", 100, 0)]], RedactConfig())
    assert not plan.ok and plan.boxes == {}


def test_plan_fails_closed_when_document_missing():
    cfg = RedactConfig(documents=[DocumentKind("bill", ["widget"]), DocumentKind("receipt", ["receipt"])])
    plan = plan_redaction([_bill()], cfg)
    assert not plan.ok and "receipt" in plan.reason


def test_summary_hides_values_by_default():
    plan = plan_redaction([_bill()], RedactConfig())
    assert "values" not in plan.summary()
    assert 50.0 in plan.summary(include_values=True)["values"]


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"amount": {}})
