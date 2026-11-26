"""Хранилище результатов (SQLite или DWH через SQLAlchemy)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine,
                        delete, func, select)

from .engine import STATE_COLS

metadata = MetaData()

T_CONTRACT_GROUP = Table(
    "contract_group", metadata,
    Column("contract_id", String(64), primary_key=True),
    Column("group_id", String(128), index=True),
    Column("portfolio", String(64)), Column("cohort", Integer), Column("profitability", String(16)),
    Column("recog_period", String(7), index=True),
    Column("premium", Float), Column("acquisition_cost", Float),
    Column("bel_init", Float), Column("ra_init", Float), Column("csm_init", Float), Column("margin_pct", Float),
)
T_GROUP_STATE = Table(
    "group_state", metadata,
    Column("period", String(7), primary_key=True), Column("group_id", String(128), primary_key=True),
    *[Column(c, Float) for c in STATE_COLS],
)
T_GROUP_DETAIL = Table(
    "group_detail", metadata,
    Column("period", String(7), index=True), Column("contract_id", String(64)), Column("group_id", String(128)),
    Column("bel", Float), Column("bel_locked", Float), Column("ra", Float), Column("status", String(24)),
)
T_REPORT_LINES = Table(
    "report_lines", metadata,
    Column("period", String(7), index=True), Column("group_id", String(128)), Column("report", String(16)),
    Column("line", String(32)), Column("col", String(16)), Column("value", Float),
)
T_CHECKS = Table(
    "checks", metadata,
    Column("period", String(7), index=True), Column("group_id", String(128)), Column("check", String(128)),
    Column("diff", Float), Column("passed", Boolean), Column("severity", String(16)), Column("info", String(256)),
)
T_RUN_LOG = Table(
    "run_log", metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("period", String(7)), Column("started_at", DateTime), Column("finished_at", DateTime),
    Column("status", String(16)), Column("n_contracts", Integer), Column("n_new", Integer),
    Column("checks_failed", Integer), Column("assumptions", String(64)), Column("config_hash", String(32)),
    Column("message", String(1000)),
)
PERIOD_TABLES = [T_GROUP_STATE, T_GROUP_DETAIL, T_REPORT_LINES, T_CHECKS]


class ResultStore:
    def __init__(self, url: str, schema: str | None = None):
        self.url = url
        self.schema = schema
        self.engine = create_engine(url)
        if schema:
            for t in metadata.tables.values():
                t.schema = schema
        metadata.create_all(self.engine)

    # --- чтение ----------------------------------------------------------
    def _read(self, table: Table, where=None) -> pd.DataFrame:
        stmt = select(table)
        if where is not None:
            stmt = stmt.where(where)
        with self.engine.connect() as con:
            return pd.read_sql(stmt, con)

    def last_closed_period(self) -> str | None:
        with self.engine.connect() as con:
            return con.execute(select(func.max(T_GROUP_STATE.c.period))).scalar()

    def first_closed_period(self) -> str | None:
        with self.engine.connect() as con:
            return con.execute(select(func.min(T_GROUP_STATE.c.period))).scalar()

    def closed_periods(self) -> list[str]:
        with self.engine.connect() as con:
            rows = con.execute(select(T_GROUP_STATE.c.period).distinct().order_by(T_GROUP_STATE.c.period)).all()
        return [r[0] for r in rows]

    def group_state(self, period: str) -> pd.DataFrame:
        return self._read(T_GROUP_STATE, T_GROUP_STATE.c.period == period)

    def contract_groups(self) -> pd.DataFrame:
        return self._read(T_CONTRACT_GROUP)

    def group_detail(self, period: str) -> pd.DataFrame:
        return self._read(T_GROUP_DETAIL, T_GROUP_DETAIL.c.period == period)

    def report_lines(self, p_from: str, p_to: str) -> pd.DataFrame:
        return self._read(T_REPORT_LINES, (T_REPORT_LINES.c.period >= p_from) & (T_REPORT_LINES.c.period <= p_to))

    def checks(self, period: str | None = None) -> pd.DataFrame:
        return self._read(T_CHECKS, T_CHECKS.c.period == period if period else None)

    def run_log(self) -> pd.DataFrame:
        df = self._read(T_RUN_LOG)
        return df.sort_values("run_id") if len(df) else df

    # --- запись ----------------------------------------------------------
    def delete_from(self, period: str) -> None:
        """Удаляет результаты месяца `period` и всех последующих (перезапуск)."""
        with self.engine.begin() as con:
            for t in PERIOD_TABLES:
                con.execute(delete(t).where(t.c.period >= period))
            con.execute(delete(T_CONTRACT_GROUP).where(T_CONTRACT_GROUP.c.recog_period >= period))

    def save_period(self, res) -> None:
        with self.engine.begin() as con:
            for t in PERIOD_TABLES:
                con.execute(delete(t).where(t.c.period == res.period))
            con.execute(delete(T_CONTRACT_GROUP).where(T_CONTRACT_GROUP.c.recog_period == res.period))
            for t, df in [(T_GROUP_STATE, res.group_state), (T_GROUP_DETAIL, res.group_detail),
                          (T_REPORT_LINES, res.report_lines), (T_CHECKS, res.checks),
                          (T_CONTRACT_GROUP, res.new_contract_groups)]:
                if df is not None and len(df):
                    recs = df[[c.name for c in t.columns]].to_dict("records")
                    con.execute(t.insert(), recs)

    def load_opening_state(self, period: str, df: pd.DataFrame) -> None:
        """Загрузка входящих остатков (например, на дату перехода) в group_state."""
        df = df.copy()
        df["period"] = period
        for c in STATE_COLS:
            if c not in df:
                df[c] = 0.0
        with self.engine.begin() as con:
            con.execute(delete(T_GROUP_STATE).where(T_GROUP_STATE.c.period == period))
            con.execute(T_GROUP_STATE.insert(), df[[c.name for c in T_GROUP_STATE.columns]].to_dict("records"))

    def log_run(self, period: str, started: dt.datetime, status: str, n_contracts: int = 0, n_new: int = 0,
                checks_failed: int = 0, assumptions: str = "", config_hash: str = "", message: str = "") -> None:
        with self.engine.begin() as con:
            con.execute(T_RUN_LOG.insert(), [{
                "period": period, "started_at": started, "finished_at": dt.datetime.now(), "status": status,
                "n_contracts": n_contracts, "n_new": n_new, "checks_failed": checks_failed,
                "assumptions": assumptions, "config_hash": config_hash, "message": message[:1000],
            }])
