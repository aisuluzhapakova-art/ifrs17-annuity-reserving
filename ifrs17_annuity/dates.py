"""Работа с месяцами.

Месяц кодируется целым индексом idx = год*12 + (месяц-1).
Оценка выполняется на конец месяца idx.
"""
from __future__ import annotations

import calendar
import datetime as dt

import numpy as np
import pandas as pd


def period_to_idx(period: str) -> int:
    """'2026-11' -> индекс месяца."""
    s = str(period).strip()
    y, m = s[:7].split("-")
    return int(y) * 12 + int(m) - 1


def idx_to_period(idx: int) -> str:
    idx = int(idx)
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def month_end(idx: int) -> dt.date:
    y, m = int(idx) // 12, int(idx) % 12 + 1
    return dt.date(y, m, calendar.monthrange(y, m)[1])


def month_start(idx: int) -> dt.date:
    return dt.date(int(idx) // 12, int(idx) % 12 + 1, 1)


def date_to_idx(d) -> int:
    d = pd.Timestamp(d)
    return d.year * 12 + d.month - 1


def series_to_idx(s: pd.Series, missing: int = 10**7) -> np.ndarray:
    """Даты -> индексы месяцев; пустые значения -> `missing` (бесконечное будущее)."""
    s = pd.to_datetime(s, errors="coerce")
    out = (s.dt.year * 12 + s.dt.month - 1).to_numpy(dtype="float64")
    out = np.where(np.isnan(out), missing, out)
    return out.astype(np.int64)


def period_range(p_from: str, p_to: str) -> list[str]:
    a, b = period_to_idx(p_from), period_to_idx(p_to)
    return [idx_to_period(i) for i in range(a, b + 1)]


def year_start_idx(idx: int) -> int:
    return (int(idx) // 12) * 12
