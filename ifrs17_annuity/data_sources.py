"""Чтение исходных данных (DWH через SQLAlchemy или CSV-выгрузка) и проверки качества данных."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

CONTRACT_COLS = ["contract_id", "product_code", "conclusion_date", "payment_start_date", "birth_date", "gender",
                 "single_premium", "initial_monthly_annuity", "indexation_rate", "guarantee_months",
                 "acquisition_cost", "status", "death_date", "termination_date", "termination_reason", "currency"]
REQUIRED_CONTRACT = ["contract_id", "product_code", "conclusion_date", "payment_start_date", "birth_date",
                     "gender", "single_premium", "initial_monthly_annuity"]
CF_COLS = ["contract_id", "cf_type", "due_date", "paid_date", "amount"]
CURVE_COLS = ["curve_date", "tenor_years", "zero_rate"]
CF_TYPES = {"PREMIUM", "COMMISSION", "ANNUITY", "SURRENDER", "EXPENSE"}
DATE_COLS = ["conclusion_date", "payment_start_date", "birth_date", "death_date", "termination_date"]


@dataclass
class SourceData:
    contracts: pd.DataFrame
    cashflows: pd.DataFrame
    curve: pd.DataFrame
    issues: list[dict] = field(default_factory=list)

    @property
    def errors(self) -> list[dict]:
        return [i for i in self.issues if i["severity"] == "ERROR"]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"contract_id": str})


def _normalize(contracts: pd.DataFrame, cashflows: pd.DataFrame, curve: pd.DataFrame):
    contracts = contracts.rename(columns=str.lower)
    cashflows = cashflows.rename(columns=str.lower)
    curve = curve.rename(columns=str.lower)
    for c in CONTRACT_COLS:
        if c not in contracts:
            contracts[c] = None
    for c in DATE_COLS:
        contracts[c] = pd.to_datetime(contracts[c], errors="coerce")
    contracts["contract_id"] = contracts["contract_id"].astype(str)
    for c in ["single_premium", "initial_monthly_annuity", "indexation_rate", "acquisition_cost"]:
        contracts[c] = pd.to_numeric(contracts[c], errors="coerce")
    contracts["guarantee_months"] = pd.to_numeric(contracts["guarantee_months"], errors="coerce").fillna(0).astype(int)
    contracts["indexation_rate"] = contracts["indexation_rate"].fillna(0.0)
    contracts["acquisition_cost"] = contracts["acquisition_cost"].fillna(0.0)
    contracts["gender"] = contracts["gender"].astype(str).str.strip().str.upper()
    for c in ["status", "termination_reason", "product_code"]:
        contracts[c] = contracts[c].where(contracts[c].isna(), contracts[c].astype(str).str.strip().str.upper())

    if len(cashflows):
        cashflows["contract_id"] = cashflows["contract_id"].astype(str)
        cashflows["cf_type"] = cashflows["cf_type"].astype(str).str.strip().str.upper()
        cashflows["due_date"] = pd.to_datetime(cashflows["due_date"], errors="coerce")
        cashflows["paid_date"] = pd.to_datetime(cashflows["paid_date"], errors="coerce")
        cashflows["amount"] = pd.to_numeric(cashflows["amount"], errors="coerce")
    curve["curve_date"] = pd.to_datetime(curve["curve_date"], errors="coerce")
    curve["tenor_years"] = pd.to_numeric(curve["tenor_years"], errors="coerce")
    curve["zero_rate"] = pd.to_numeric(curve["zero_rate"], errors="coerce")
    return contracts, cashflows, curve


def load_source(cfg: Config, upto=None) -> SourceData:
    """Загрузка витрин. `upto` — дата отсечения потоков (включительно), если задана."""
    src = cfg.source
    views = src.get("views", {})
    typ = str(src.get("type", "csv")).lower()
    if typ == "csv":
        d = cfg.resolve(src.get("csv_dir", "data/demo"))
        contracts = _read_csv(d / f"{views.get('contracts', 'v_annuity_contracts')}.csv")
        cashflows = _read_csv(d / f"{views.get('cashflows', 'v_annuity_cashflows')}.csv")
        curve = pd.read_csv(d / f"{views.get('curve', 'v_discount_curve')}.csv")
    elif typ == "dwh":
        from sqlalchemy import create_engine, text

        url = os.environ.get(src.get("dwh_url_env", "DWH_URL"))
        if not url:
            raise RuntimeError(f"Не задана переменная окружения {src.get('dwh_url_env', 'DWH_URL')} с URL DWH")
        schema = src.get("dwh_schema")
        q = lambda v: f"{schema}.{v}" if schema else v  # noqa: E731
        eng = create_engine(url)
        with eng.connect() as con:
            contracts = pd.read_sql(text(f"SELECT * FROM {q(views['contracts'])}"), con)
            if upto is not None:
                cashflows = pd.read_sql(text(f"SELECT * FROM {q(views['cashflows'])} WHERE due_date <= :d"),
                                        con, params={"d": pd.Timestamp(upto).to_pydatetime()})
            else:
                cashflows = pd.read_sql(text(f"SELECT * FROM {q(views['cashflows'])}"), con)
            curve = pd.read_sql(text(f"SELECT * FROM {q(views['curve'])}"), con)
    else:
        raise ValueError(f"Неизвестный тип источника: {typ}")

    contracts, cashflows, curve = _normalize(contracts, cashflows, curve)
    sd = SourceData(contracts=contracts, cashflows=cashflows, curve=curve)
    sd.issues = run_dq(sd, cfg)
    return sd


def run_dq(sd: SourceData, cfg: Config) -> list[dict]:
    """Проверки качества данных (раздел 6.2). ERROR блокирует расчет, WARNING — нет."""
    issues: list[dict] = []

    def add(sev, check, n, msg, ids=None):
        if n:
            issues.append({"severity": sev, "check": check, "count": int(n), "message": msg,
                           "examples": ", ".join(map(str, list(ids)[:5])) if ids is not None else ""})

    c, f, k = sd.contracts, sd.cashflows, sd.curve
    for col in REQUIRED_CONTRACT:
        if col not in c.columns:
            add("ERROR", "Обязательные колонки", 1, f"Нет колонки {col} в витрине договоров")
            continue
        bad = c[c[col].isna()]
        add("ERROR", "Обязательные поля", len(bad), f"Пустое поле {col}", bad["contract_id"])
    dup = c[c["contract_id"].duplicated(keep=False)]
    add("ERROR", "Уникальность contract_id", dup["contract_id"].nunique(), "Дублирующиеся contract_id", dup["contract_id"].unique())
    bad = c[~c["gender"].isin(["M", "F"])]
    add("ERROR", "Пол", len(bad), "Пол должен быть M или F", bad["contract_id"])
    bad = c[c["product_code"].notna() & ~c["product_code"].isin(list(cfg.portfolios))]
    add("ERROR", "Портфель", len(bad), "Продукт без маппинга в портфель: " + ", ".join(sorted(bad["product_code"].astype(str).unique())), bad["contract_id"])
    bad = c[(c["single_premium"] <= 0) | (c["initial_monthly_annuity"] <= 0)]
    add("ERROR", "Суммы договора", len(bad), "Премия и аннуитет должны быть > 0", bad["contract_id"])
    bad = c[c["payment_start_date"] < c["conclusion_date"]]
    add("WARNING", "Даты", len(bad), "Дата первой выплаты раньше даты заключения", bad["contract_id"])
    bad = c[(c["status"] == "ACTIVE") & c["termination_date"].notna()]
    add("WARNING", "Статус", len(bad), "Договор ACTIVE с датой прекращения", bad["contract_id"])
    bad = c[(c["death_date"].notna()) & (c["death_date"] < c["birth_date"])]
    add("ERROR", "Даты", len(bad), "Дата смерти раньше даты рождения", bad["contract_id"])

    for col in CF_COLS:
        if col not in f.columns:
            add("ERROR", "Обязательные колонки", 1, f"Нет колонки {col} в витрине потоков")
    if len(f) and set(CF_COLS) <= set(f.columns):
        bad = f[~f["cf_type"].isin(CF_TYPES)]
        add("ERROR", "Типы потоков", len(bad), "Неизвестный cf_type: " + ", ".join(sorted(bad["cf_type"].astype(str).unique())), bad["contract_id"])
        bad = f[(f["amount"] < 0) | f["amount"].isna()]
        add("ERROR", "Суммы потоков", len(bad), "Сумма потока должна быть ≥ 0", bad["contract_id"])
        bad = f[f["due_date"].isna()]
        add("ERROR", "Даты потоков", len(bad), "Пустая due_date", bad["contract_id"])
        unknown = set(f["contract_id"]) - set(c["contract_id"])
        add("WARNING", "Ссылочная целостность", len(unknown), "Потоки по неизвестным договорам", sorted(unknown))
        prem = f[f["cf_type"] == "PREMIUM"].groupby("contract_id")["amount"].sum()
        cp = c.set_index("contract_id")["single_premium"]
        joined = pd.concat([cp, prem.rename("dwh")], axis=1, join="inner")
        bad = joined[(joined["single_premium"] - joined["dwh"]).abs() > 1.0]
        add("WARNING", "Премия", len(bad), "Премия в договоре ≠ поступлению в DWH", bad.index)

    for col in CURVE_COLS:
        if col not in k.columns:
            add("ERROR", "Обязательные колонки", 1, f"Нет колонки {col} в витрине кривой")
    if len(k) == 0:
        add("ERROR", "Кривая", 1, "Витрина кривой пуста")
    return issues
