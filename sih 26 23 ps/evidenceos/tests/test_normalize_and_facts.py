"""Unit tests for the deterministic parts of EvidenceOS (no OCR / models needed).

    python -m pytest -q tests/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidenceos.docai.normalize import normalize_unit, parse_number, parse_period, to_display  # noqa: E402
from evidenceos.knowledge.facts import DocContext, classify_columns, clean_header, extract_cell_value, lead_in_sentence  # noqa: E402
from evidenceos.docai.layout import is_aggregate_label, value_like  # noqa: E402


def test_parse_number_indian_grouping_and_markers():
    assert parse_number("3,89,421.34") == 389421.34
    assert parse_number("1,047.68") == 1047.68
    assert parse_number("(0.5)") == -0.5
    assert extract_cell_value("▼ 1.45")[0] == -1.45
    assert extract_cell_value("▲ 2.01")[0] == 2.01
    assert extract_cell_value("\uf03510.02%")[0] == 10.02


def test_units_keep_original_and_canonicalise():
    u, mult, orig = normalize_unit("Million Tonnes")
    assert u == "tonnes" and mult == 1e6
    assert to_display(1047.68e6, "tonnes") == "1,047.68 MT"
    u, mult, _ = normalize_unit("M.Cum")
    assert u == "cubic_metres" and mult == 1e6


def test_periods():
    assert parse_period("FY 25").key == "FY2024-25"
    assert parse_period("2024-25").key == "FY2024-25"
    assert parse_period("Mar'25").key == "2025-03"
    assert parse_period("Upto Mar'25").key == "FY2024-25"           # April–March cumulative = full year
    assert parse_period("2025-26 (upto December 25)").key == "FY2025-26:ytd:2025-12"
    assert parse_period("as on 01.04.2024").key == "asof:2024-04-01"
    p = parse_period("Projected Production (Jan- Mar'26)")
    assert p.scope == "range" and p.key.endswith("2026-01..2026-03")


def test_clean_header_wraps_but_keeps_month_ranges():
    assert clean_header("Project- ed Pro- duction (Jan- Mar’25)") == "Projected Production (Jan- Mar'25)"
    assert clean_header("2023- 24 Actual") == "2023-24 Actual"


def test_value_like_rejects_header_artefacts():
    assert not value_like("6.", None)          # list numbering
    assert not value_like("(331)", None)       # resource category code
    assert not value_like("24", "2023-")       # wrapped fiscal year
    assert value_like("1047.68", None)
    assert is_aggregate_label("Grand Total") and not is_aggregate_label("Captive & Others")


def test_classify_monthly_statistics_header():
    headers = ["Sl No", "Subs", "Monthly Target", "Production during Mar FY 25", "Achmt.(%)", "Production during Mar FY 24", "Growth (%)",
               "Production upto Mar FY 25", "Production upto Mar FY 24", "Growth (%)"]
    rows = [["1", "ECL", "6.60", "7.07", "106.98", "6.93", "▲ 2.01", "52.04", "47.56", "▲ 9.41"]]
    ctx = DocContext(document_id="d", doc_code="D", doc_type="monthly_statistics", period="2025-03", title="Monthly Coal Statistics March 2025 (Provisional)")
    specs, ent_col, parent_col, _ = classify_columns(headers, rows, "", "mt", ctx)
    by = {s.index: s for s in specs}
    assert ent_col == 1 and parent_col is None
    assert by[3].predicate == "production" and by[3].period.key == "2025-03" and by[3].qualifier == "provisional"
    assert by[7].period.key == "FY2024-25" and by[8].period.key == "FY2023-24"
    assert by[4].kind == "percent" and by[4].predicate.endswith("achievement_pct")
    assert by[9].kind == "percent" and by[9].period.key == "FY2024-25"


def test_classify_annual_report_header_with_ytd_group():
    headers = ["Company", "2023- 24 Actual", "2024-25 Annual Target", "2024-25 Actual", "Ach. (%)", "Growth (%)",
               "2025-26 (upto December 25) Target", "2025-26 (upto December 25) Actual", "2025-26 (upto December 25) Ach. (%)", "Projected Production (Jan- Mar’26)"]
    rows = [["CIL", "773.7", "838", "781.06", "93.18%", "0.96", "605.38", "529.19", "87.41%", "346.05"]]
    ctx = DocContext(document_id="d", doc_code="D", doc_type="annual_report", period="FY2025-26", title="Annual Report 2025-26")
    specs, ent_col, _, _ = classify_columns(headers, rows, "COMPANY WISE COAL PRODUCTION [in Million Tonne (MT)]", "Million Tonne", ctx)
    by = {s.index: s for s in specs}
    assert by[1].period.key == "FY2023-24" and by[2].qualifier == "target" and by[3].period.key == "FY2024-25"
    assert by[6].period.key == "FY2025-26:ytd:2025-12" and by[6].qualifier == "target"
    assert by[9].qualifier == "projected" and by[9].period.scope == "range"


def test_lead_in_sentence_accepts_table_titles_and_colons():
    assert lead_in_sentence("Some paragraph.\nTable 3.11 : Company Wise Production of Coal in last Three Years").startswith("Table 3.11")
    assert lead_in_sentence("Sector-wise coal off-take from CIL during the period from Jan’24-Nov’24 is as below:-").endswith(":-")
    assert lead_in_sentence("This is an unrelated paragraph that ends with a full stop.") == ""
