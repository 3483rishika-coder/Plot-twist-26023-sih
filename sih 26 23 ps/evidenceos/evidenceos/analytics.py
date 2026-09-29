"""Tabular Analytics & DataFrame Engine (PRD §8, §11.4).

Utilizes pandas for:
- Converting extracted document tables into structured pandas DataFrames.
- Multi-format table export: CSV, Excel (.xlsx), Parquet, and JSON records.
- Vectorized cross-table analytics: YoY growth variance, Target vs Actual achievement pivots,
  and subsidiary production comparison matrices.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional

import pandas as pd

from .db import Cell, Document, Fact, Table

logger = logging.getLogger(__name__)


def table_to_dataframe(session, table_id: str) -> pd.DataFrame:
    """Convert an EvidenceOS extracted table into a clean pandas DataFrame."""
    t = session.get(Table, table_id)
    if not t:
        return pd.DataFrame()

    cells = session.query(Cell).filter(Cell.table_id == table_id).order_by(Cell.row_index, Cell.col_index).all()
    if not cells:
        return pd.DataFrame()

    max_row = max(c.row_index for c in cells)
    max_col = max(c.col_index for c in cells)

    # Initialize empty grid
    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for c in cells:
        grid[c.row_index][c.col_index] = c.text or ""

    if not grid:
        return pd.DataFrame()

    headers = [h or f"Col_{i}" for i, h in enumerate(grid[0])]
    data = grid[1:] if len(grid) > 1 else []
    df = pd.DataFrame(data, columns=headers)

    # Clean numeric columns where possible
    for col in df.columns:
        cleaned = df[col].astype(str).str.replace(",", "").str.strip()
        numeric_series = pd.to_numeric(cleaned, errors="coerce")
        if numeric_series.notna().sum() > (len(df) * 0.4):
            df[col] = numeric_series

    return df


def export_table(session, table_id: str, fmt: str = "csv") -> tuple[bytes, str, str]:
    """Export table in requested format: 'csv', 'xlsx', 'json', or 'parquet'.
    Returns (content_bytes, media_type, filename_extension).
    """
    df = table_to_dataframe(session, table_id)
    fmt = fmt.lower().strip()

    if fmt == "xlsx" or fmt == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Extracted Table")
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"

    elif fmt == "parquet":
        buf = io.BytesIO()
        # Convert all object columns to strings for safe parquet serialization
        df.astype(str).to_parquet(buf, index=False)
        return buf.getvalue(), "application/octet-stream", "parquet"

    elif fmt == "json":
        buf = io.BytesIO()
        df.to_json(buf, orient="records", indent=2)
        return buf.getvalue(), "application/json", "json"

    else:  # default to CSV
        buf = io.BytesIO()
        df.to_csv(buf, index=False, encoding="utf-8")
        return buf.getvalue(), "text/csv", "csv"


def facts_to_production_matrix(session, period: Optional[str] = None) -> dict:
    """Generate a pandas pivot table of subsidiary production and despatch."""
    q = session.query(Fact).filter(Fact.predicate.in_(["production", "dispatch", "overburden"]))
    if period:
        q = q.filter(Fact.period == period)

    facts = q.all()
    if not facts:
        return {"period": period, "columns": [], "rows": [], "summary": "No facts available"}

    rows_data = []
    for f in facts:
        if f.value is not None:
            val_mt = round(f.value / 1e6, 2) if f.unit == "tonnes" else round(f.value, 2)
            rows_data.append({
                "subsidiary": f.subject_text,
                "metric": f"{f.predicate}_{f.qualifier or 'actual'}",
                "period": f.period,
                "value_mt": val_mt,
            })

    if not rows_data:
        return {"period": period, "columns": [], "rows": [], "summary": "No numeric facts"}

    df = pd.DataFrame(rows_data)

    # Perform pandas pivot table
    pivot = pd.pivot_table(
        df,
        values="value_mt",
        index=["subsidiary"],
        columns=["metric"],
        aggfunc="max",
        fill_value=0.0,
    ).reset_index()

    # Calculate summary metrics using pandas
    cols = [c for c in pivot.columns if c != "subsidiary"]
    totals = {"subsidiary": "Total (Vectorized Sum)"}
    for c in cols:
        totals[c] = round(float(pivot[c].sum()), 2)

    rows = pivot.to_dict(orient="records")
    rows.append(totals)

    return {
        "period": period or "all",
        "metrics": cols,
        "subsidiaries_count": len(pivot),
        "data": rows,
    }
