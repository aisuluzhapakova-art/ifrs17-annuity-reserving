"""Конфигурация: учетная политика, портфели, параметры кривой, версии допущений."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

from .dates import date_to_idx, month_end, period_to_idx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class MortalityTable:
    """qx по полу и возрасту: массив (2, max_age+1); 0 = M, 1 = F."""
    qx: np.ndarray

    @classmethod
    def from_csv(cls, path: Path, max_age: int) -> "MortalityTable":
        df = pd.read_csv(path)
        df = df.set_index("age").sort_index()
        ages = np.arange(max_age + 1)
        q = np.ones((2, max_age + 1))
        for g, col in enumerate(["qx_M", "qx_F"]):
            s = df[col].reindex(ages)
            s = s.ffill().fillna(1.0)
            q[g] = np.clip(s.to_numpy(dtype=float), 0.0, 1.0)
        q[:, max_age] = 1.0
        return cls(qx=q)


@dataclass
class AssumptionSet:
    version: str
    effective_from: pd.Timestamp
    mortality_table: MortalityTable
    mortality_table_path: str
    base_year: int
    experience_factor: dict  # {'M': 1.0, 'F': 1.0}
    improvement: float
    per_policy_monthly: float
    pct_of_payments: float
    inflation: float
    expense_base_idx: int
    ra_confidence: float
    mortality_shock_995: float
    expense_shock_995: float

    @property
    def ra_scale(self) -> float:
        """Масштаб шока 99,5% к целевому уровню доверия: Φ⁻¹(CL)/Φ⁻¹(0,995)."""
        nd = NormalDist()
        return nd.inv_cdf(self.ra_confidence) / nd.inv_cdf(0.995)

    @property
    def mortality_shock(self) -> float:
        return self.mortality_shock_995 * self.ra_scale

    @property
    def expense_shock(self) -> float:
        return self.expense_shock_995 * self.ra_scale

    def as_dict(self) -> dict:
        return {
            "Версия": self.version,
            "Действует с": str(self.effective_from.date()),
            "Таблица смертности": self.mortality_table_path,
            "Базовый год таблицы": self.base_year,
            "Коэффициент опыта M": self.experience_factor.get("M", 1.0),
            "Коэффициент опыта F": self.experience_factor.get("F", 1.0),
            "Improvement, в год": self.improvement,
            "Расходы на договор в месяц, KZT": self.per_policy_monthly,
            "Расходы, % от выплат": self.pct_of_payments,
            "Инфляция расходов": self.inflation,
            "RA: уровень доверия": self.ra_confidence,
            "RA: шок смертности 99,5%": self.mortality_shock_995,
            "RA: шок расходов 99,5%": self.expense_shock_995,
            "RA: масштаб шока": round(self.ra_scale, 6),
            "RA: шок смертности (применяемый)": round(self.mortality_shock, 6),
            "RA: шок расходов (применяемый)": round(self.expense_shock, 6),
        }


@dataclass
class Config:
    raw: dict
    path: Path
    root: Path
    assumptions: list[AssumptionSet] = field(default_factory=list)

    # --- удобные свойства -------------------------------------------------
    @property
    def entity(self) -> dict:
        return self.raw.get("entity", {})

    @property
    def units(self) -> float:
        return float(self.entity.get("report_units", 1000))

    @property
    def policy(self) -> dict:
        return self.raw["policy"]

    @property
    def portfolios(self) -> dict:
        return self.raw["portfolios"]

    @property
    def discount(self) -> dict:
        return self.raw["discount"]

    @property
    def source(self) -> dict:
        return self.raw["source"]

    @property
    def max_age(self) -> int:
        return int(self.policy.get("max_age", 110))

    @property
    def tolerance(self) -> float:
        return float(self.policy.get("tolerance", 0.5))

    @property
    def ifie_policy(self) -> str:
        return str(self.policy.get("ifie_disaggregation", "OCI")).upper()

    @property
    def ra_disaggregation(self) -> bool:
        return bool(self.policy.get("ra_disaggregation", False))

    @property
    def guarantee_as_ic(self) -> bool:
        return bool(self.policy.get("guarantee_as_investment_component", True))

    @property
    def no_sig_threshold(self) -> float:
        return float(self.policy.get("no_sig_risk_threshold", 0.06))

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.raw.get("output_dir", "output"))

    @property
    def results_url(self) -> str:
        url = os.environ.get("RESULTS_URL") or self.raw["results_store"]["url"]
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
            rel = url[len("sqlite:///"):]
            if rel and rel != ":memory:" and not Path(rel).is_absolute():
                p = self.resolve(rel)
                p.parent.mkdir(parents=True, exist_ok=True)
                url = "sqlite:///" + p.as_posix()
        return url

    @property
    def results_schema(self):
        return self.raw["results_store"].get("schema")

    def resolve(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)

    def assumptions_at(self, idx: int) -> AssumptionSet:
        """Версия допущений, действующая на конец месяца idx."""
        d = pd.Timestamp(month_end(idx))
        valid = [a for a in self.assumptions if a.effective_from <= d]
        if not valid:
            return self.assumptions[0]
        return max(valid, key=lambda a: a.effective_from)

    def portfolio_of(self, product_code: str) -> str | None:
        return self.portfolios.get(product_code)

    def fingerprint(self) -> str:
        s = json.dumps(self.raw, sort_keys=True, default=str)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

    def with_overrides(self, **policy) -> "Config":
        raw = copy.deepcopy(self.raw)
        raw["policy"].update(policy)
        return Config.from_dict(raw, self.path, self.root)

    # --- загрузка ---------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict, path: Path | None = None, root: Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG
        root = Path(root) if root else PROJECT_ROOT
        cfg = cls(raw=raw, path=path, root=root)
        max_age = cfg.max_age
        tables: dict[str, MortalityTable] = {}
        for a in raw["assumptions"]:
            mt = a["mortality"]
            tpath = str(mt["table"])
            if tpath not in tables:
                tables[tpath] = MortalityTable.from_csv(cfg.resolve(tpath), max_age)
            ex = a["expenses"]
            ra = a["ra"]
            cfg.assumptions.append(AssumptionSet(
                version=str(a["version"]),
                effective_from=pd.Timestamp(a["effective_from"]),
                mortality_table=tables[tpath],
                mortality_table_path=tpath,
                base_year=int(mt.get("base_year", 2024)),
                experience_factor={k: float(v) for k, v in mt.get("experience_factor", {}).items()},
                improvement=float(mt.get("improvement", 0.0)),
                per_policy_monthly=float(ex.get("per_policy_monthly", 0.0)),
                pct_of_payments=float(ex.get("pct_of_payments", 0.0)),
                inflation=float(ex.get("inflation", 0.0)),
                expense_base_idx=period_to_idx(str(ex.get("base_period", "2024-01"))),
                ra_confidence=float(ra.get("confidence_level", 0.75)),
                mortality_shock_995=float(ra.get("mortality_shock_995", 0.0)),
                expense_shock_995=float(ra.get("expense_shock_995", 0.0)),
            ))
        cfg.assumptions.sort(key=lambda a: a.effective_from)
        if not cfg.assumptions:
            raise ValueError("В конфигурации нет ни одной версии допущений")
        if cfg.ifie_policy not in ("OCI", "PL"):
            raise ValueError("policy.ifie_disaggregation должен быть OCI или PL")
        return cfg


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    path = path.resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    root = path.parent.parent if path.parent.name == "config" else path.parent
    return Config.from_dict(raw, path, root)


__all__ = ["Config", "AssumptionSet", "MortalityTable", "load_config", "date_to_idx", "PROJECT_ROOT"]
