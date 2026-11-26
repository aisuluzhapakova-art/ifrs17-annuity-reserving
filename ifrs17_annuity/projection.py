"""Поконтрактная помесячная проекция денежных потоков (векторизованная, numpy).

Месяц m = T + k, k = 1..N, где T — месяц оценки (конец месяца).
Аннуитет за месяц m: A0·(1+i)^⌊(m−S)/12⌋ при m ≥ S. Выплачивается, если m ≤ конца
гарантированного периода или получатель жив на конец m (FR-9).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AssumptionSet
from .dates import series_to_idx

FAR = 10**7


@dataclass
class PortfolioArrays:
    contract_id: np.ndarray
    birth_idx: np.ndarray
    gender: np.ndarray        # 0 = M, 1 = F
    a0: np.ndarray            # первая ежемесячная выплата
    ind: np.ndarray           # годовая индексация
    s_idx: np.ndarray         # месяц первой выплаты
    g_end: np.ndarray         # последний месяц гарантированного периода (S-1, если гарантии нет)
    death_idx: np.ndarray     # месяц смерти (FAR, если жив)
    cancel_idx: np.ndarray    # месяц расторжения/перевода (FAR, если нет)

    def __len__(self) -> int:
        return len(self.contract_id)

    def subset(self, mask) -> "PortfolioArrays":
        return PortfolioArrays(**{k: v[mask] for k, v in self.__dict__.items()})

    def with_a0(self, a0) -> "PortfolioArrays":
        d = dict(self.__dict__)
        d["a0"] = np.broadcast_to(np.asarray(a0, float), self.a0.shape).copy()
        return PortfolioArrays(**d)

    @classmethod
    def from_contracts(cls, c: pd.DataFrame) -> "PortfolioArrays":
        s_idx = series_to_idx(c["payment_start_date"])
        g = c["guarantee_months"].fillna(0).astype(int).to_numpy()
        reason = c.get("termination_reason", pd.Series([None] * len(c), index=c.index)).fillna("").astype(str).str.upper()
        canc = series_to_idx(c["termination_date"], FAR) if "termination_date" in c else np.full(len(c), FAR)
        canc = np.where(reason.isin(["CANCELLATION", "TRANSFER"]).to_numpy(), canc, FAR)
        return cls(
            contract_id=c["contract_id"].astype(str).to_numpy(),
            birth_idx=series_to_idx(c["birth_date"]),
            gender=(c["gender"].astype(str).str.upper() == "F").astype(np.int64).to_numpy(),
            a0=c["initial_monthly_annuity"].astype(float).to_numpy(),
            ind=c["indexation_rate"].fillna(0).astype(float).to_numpy(),
            s_idx=s_idx,
            g_end=s_idx + g - 1,
            death_idx=series_to_idx(c["death_date"], FAR) if "death_date" in c else np.full(len(c), FAR),
            cancel_idx=canc,
        )


@dataclass
class Projection:
    ann_nonic: np.ndarray   # ожидаемые аннуитетные выплаты, не являющиеся ИК (n×N)
    ann_ic: np.ndarray      # инвестиционный компонент (гарантированный период)
    exp: np.ndarray         # атрибутируемые расходы
    cu: np.ndarray          # единицы покрытия: ожидаемая выплата живому получателю

    @property
    def outflow(self) -> np.ndarray:
        return self.ann_nonic + self.ann_ic + self.exp


def horizon(pa: PortfolioArrays, T: int, max_age: int) -> int:
    """Число месяцев проекции, достаточное для всех договоров."""
    if len(pa) == 0:
        return 1
    to_age = max_age * 12 + 12 - (T - pa.birth_idx)
    to_g = pa.g_end - T
    return int(max(1, np.max(np.maximum(to_age, to_g)) + 1))


def project(pa: PortfolioArrays, T: int, a: AssumptionSet, max_age: int, N: int,
            guarantee_as_ic: bool = True, stressed: bool = False) -> Projection:
    n = len(pa)
    k = np.arange(1, N + 1)
    m = T + k                                  # (N,)

    alive0 = (pa.death_idx > T) & (pa.cancel_idx > T)
    active = (pa.cancel_idx > T)[:, None]

    # --- смертность (FR-10) ---
    age = (m[None, :] - pa.birth_idx[:, None]) // 12
    age_c = np.clip(age, 0, max_age)
    qx = a.mortality_table.qx[pa.gender[:, None], age_c]
    ef = np.array([a.experience_factor.get("M", 1.0), a.experience_factor.get("F", 1.0)])[pa.gender]
    year = m // 12
    impr = (1.0 - a.improvement) ** (year - a.base_year)
    qx = qx * ef[:, None] * impr[None, :]
    if stressed:
        qx = qx * (1.0 - a.mortality_shock)
    qx = np.clip(qx, 0.0, 1.0)
    qx = np.where(age >= max_age, 1.0, qx)
    qm = 1.0 - (1.0 - qx) ** (1.0 / 12.0)
    p_alive = np.cumprod(1.0 - qm, axis=1) * alive0[:, None]

    # --- аннуитет (FR-9) ---
    started = m[None, :] >= pa.s_idx[:, None]
    expo = np.where(started, (m[None, :] - pa.s_idx[:, None]) // 12, 0)
    amt = pa.a0[:, None] * np.power(1.0 + pa.ind[:, None], expo) * started
    in_g = m[None, :] <= pa.g_end[:, None]

    certain = amt * in_g * active
    life = amt * p_alive * ~in_g
    if guarantee_as_ic:
        ann_ic, ann_nonic = certain, life
    else:
        ann_ic, ann_nonic = np.zeros_like(amt), certain + life

    # --- расходы (FR-11) ---
    inforce = np.where(in_g, 1.0, p_alive) * active
    infl = (1.0 + a.inflation) ** ((m - a.expense_base_idx) / 12.0)
    exp = a.per_policy_monthly * infl[None, :] * inforce + a.pct_of_payments * (ann_ic + ann_nonic)
    if stressed:
        exp = exp * (1.0 + a.expense_shock)

    cu = amt * p_alive
    return Projection(ann_nonic=ann_nonic, ann_ic=ann_ic, exp=exp, cu=cu)


def pv(cf: np.ndarray, df: np.ndarray) -> np.ndarray:
    """PV по строкам: df — вектор (N,) или матрица (n×N)."""
    if df.ndim == 1:
        return cf @ df
    return np.einsum("ij,ij->i", cf, df)
