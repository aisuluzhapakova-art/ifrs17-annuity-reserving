"""Кривые дисконтирования (п.36, B72–B85).

Подход «снизу вверх»: бескупонная кривая ГЦБ KZT + премия за неликвидность.
До LLP — линейная интерполяция спот-ставок (годовое начисление); после LLP
форвардная интенсивность линейно сходится к ln(1+UFR) за период сходимости.

Все кривые хранятся как массивы помесячных дисконт-факторов DF[k], k = 0..L-1, DF[0] = 1.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .dates import date_to_idx, idx_to_period

log = logging.getLogger(__name__)

CURVE_MONTHS = 12 * 200  # длина массива DF (с запасом на сдвиг зафиксированной кривой)


def build_df(tenors: np.ndarray, zero_rates: np.ndarray, discount: dict,
             n_months: int = CURVE_MONTHS) -> np.ndarray:
    ilp = float(discount.get("illiquidity_premium", 0.0))
    llp = float(discount.get("llp_years", 10))
    ufr = float(discount.get("ufr", 0.06))
    conv = float(discount.get("convergence_years", 20))

    order = np.argsort(tenors)
    t = np.asarray(tenors, float)[order]
    z = np.asarray(zero_rates, float)[order] + ilp
    mask = t <= llp + 1e-9
    if mask.sum() == 0:
        raise ValueError("Нет точек кривой до LLP")
    t, z = t[mask], z[mask]
    llp_eff = min(llp, t.max())
    llp_m = int(round(llp_eff * 12))

    k = np.arange(n_months)
    tm = k / 12.0
    lndf = np.zeros(n_months)
    zt = np.interp(tm[: llp_m + 1], t, z)  # плоская экстраполяция до первой точки
    lndf[: llp_m + 1] = -tm[: llp_m + 1] * np.log1p(zt)

    # форвардная интенсивность последнего года перед LLP
    m0 = max(llp_m - 12, 0)
    f_llp = (lndf[m0] - lndf[llp_m]) / max((llp_m - m0) / 12.0, 1e-9)
    f_ufr = np.log1p(ufr)
    for j in range(llp_m + 1, n_months):
        w = min(1.0, (j / 12.0 - llp_eff) / conv) if conv > 0 else 1.0
        f = f_llp + (f_ufr - f_llp) * w
        lndf[j] = lndf[j - 1] - f / 12.0
    return np.exp(lndf)


class CurveSet:
    """Набор кривых по месяцам (конец месяца)."""

    def __init__(self, curve_points: pd.DataFrame, discount: dict):
        self.discount = discount
        self.points = curve_points.copy()
        self.points["idx"] = self.points["curve_date"].map(date_to_idx)
        self._raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for idx, g in self.points.groupby("idx"):
            self._raw[int(idx)] = (g["tenor_years"].to_numpy(float), g["zero_rate"].to_numpy(float))
        self._cache: dict[int, np.ndarray] = {}
        self.warnings: list[str] = []

    @property
    def available(self) -> list[int]:
        return sorted(self._raw)

    def source_idx(self, idx: int) -> int:
        if idx in self._raw:
            return idx
        prev = [i for i in self._raw if i <= idx]
        if not prev:
            raise ValueError(f"Нет кривой дисконтирования на {idx_to_period(idx)} и ранее")
        src = max(prev)
        msg = f"Кривая на {idx_to_period(idx)} отсутствует, используется кривая {idx_to_period(src)}"
        if msg not in self.warnings:
            self.warnings.append(msg)
            log.warning(msg)
        return src

    def df(self, idx: int) -> np.ndarray:
        """Помесячные DF текущей кривой на конец месяца idx."""
        idx = int(idx)
        if idx not in self._cache:
            src = self.source_idx(idx)
            t, z = self._raw[src]
            self._cache[idx] = build_df(t, z, self.discount)
        return self._cache[idx]

    def forward_from(self, idx: int, shift: int, n: int) -> np.ndarray:
        """DF(shift+k)/DF(shift), k=1..n — кривая месяца idx, прокатанная на `shift` месяцев."""
        d = self.df(idx)
        return d[shift + 1: shift + 1 + n] / d[shift]

    def locked_df(self, recog_idx: np.ndarray, T: int, n: int) -> np.ndarray:
        """Зафиксированные DF (B72(b)) для каждого договора: DF_τ(T-τ+k)/DF_τ(T-τ), k=1..n."""
        out = np.empty((len(recog_idx), n))
        for tau in np.unique(recog_idx):
            rows = recog_idx == tau
            a = int(T - tau)
            if a < 0:
                raise ValueError("Дата оценки раньше даты признания договора")
            out[rows] = self.forward_from(int(tau), a, n)[None, :]
        return out

    def spot_table(self, idx: int, tenors=(1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50)) -> pd.DataFrame:
        d = self.df(idx)
        rows = []
        src = self.source_idx(idx)
        t_in, z_in = self._raw[src]
        for y in tenors:
            m = int(y * 12)
            spot = d[m] ** (-1.0 / y) - 1.0
            fwd = d[m - 12] / d[m] - 1.0
            inp = float(np.interp(y, t_in, z_in)) if y <= t_in.max() else np.nan
            rows.append({"Срок, лет": y, "ГЦБ (вход)": inp, "Спот с премией": spot,
                         "Годовой форвард": fwd, "DF": d[m]})
        return pd.DataFrame(rows)
