"""Taxonomy: parsing, level accessors, formatting, matching."""

from __future__ import annotations

import pytest

from spend_classifier.taxonomy import UNSPSCCode, parse, safe_parse


def test_parse_digits_and_levels():
    c = parse("401416070003")
    assert c.digits == "401416070003"
    assert c.segment == "40"
    assert c.family == "4014"
    assert c.klass == "401416"
    assert c.commodity == "40141607"
    assert c.internal == "0003"


def test_formatting():
    c = parse("401416070003")
    assert c.standard == "40-14-16-07"
    assert c.dotted == "40-14-16-07-0003"


def test_parse_accepts_dotted_input():
    assert parse("40-14-16-07-0003").digits == "401416070003"


@pytest.mark.parametrize("bad", ["", "123", "40141607000", "abcdefghijkl", None])
def test_safe_parse_rejects_malformed(bad):
    assert safe_parse(bad) is None


def test_parse_raises_on_wrong_length():
    with pytest.raises(ValueError):
        UNSPSCCode("12345")


def test_matches_by_level():
    a = parse("401416070001")
    b = parse("401416070009")  # same commodity, different internal
    c = parse("401416090001")  # same class, different commodity
    assert a.matches(b, "commodity") is True
    assert a.matches(b, "full") is False
    assert a.matches(c, "class") is True
    assert a.matches(c, "commodity") is False
