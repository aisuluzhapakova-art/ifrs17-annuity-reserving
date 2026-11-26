"""Оркестрация закрытия месяца: данные → расчет → контроли → хранилище → Excel."""
from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .curves import CurveSet
from .data_sources import SourceData, load_source
from .dates import idx_to_period, month_end, period_range, period_to_idx
from .engine import PeriodResult, close_period, prepare_contracts
from .projection import PortfolioArrays
from .store import ResultStore

log = logging.getLogger(__name__)


class SequenceError(RuntimeError):
    pass


class DataQualityError(RuntimeError):
    pass


@dataclass
class RunSummary:
    results: list[PeriodResult] = field(default_factory=list)
    excel_path: str | None = None
    seconds: float = 0.0
    dq_issues: list[dict] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.all_passed for r in self.results)

    @property
    def failed_checks(self) -> pd.DataFrame:
        frames = [r.checks[(r.checks["severity"] == "control") & ~r.checks["passed"]] for r in self.results]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class Runner:
    def __init__(self, cfg: Config, store: ResultStore | None = None, source: SourceData | None = None):
        self.cfg = cfg
        self.store = store or ResultStore(cfg.results_url, cfg.results_schema)
        self._source = source
        self._prepared = None

    # --- данные ---------------------------------------------------------
    def source(self, upto=None) -> SourceData:
        if self._source is None:
            self._source = load_source(self.cfg, upto)
            self._prepared = None
        return self._source

    def prepared(self):
        if self._prepared is None:
            sd = self.source()
            c, f = prepare_contracts(sd.contracts, sd.cashflows, self.cfg)
            self._prepared = (c, f, PortfolioArrays.from_contracts(c), CurveSet(sd.curve, self.cfg.discount))
        return self._prepared

    # --- последовательность ---------------------------------------------
    def _check_sequence(self, period: str) -> None:
        last = self.store.last_closed_period()
        if last is None:
            return
        t, tl = period_to_idx(period), period_to_idx(last)
        if t <= tl:
            first = self.store.first_closed_period()
            log.info("Перезапуск %s: удаляются результаты %s и последующих месяцев", period, period)
            self.store.delete_from(period)
            if first is not None and t > period_to_idx(first) and self.store.last_closed_period() != idx_to_period(t - 1):
                raise SequenceError(f"Нет закрытого месяца {idx_to_period(t - 1)}")
            return
        if t != tl + 1:
            raise SequenceError(f"Последний закрытый месяц {last}. Сначала закройте {idx_to_period(tl + 1)}")

    # --- расчет ---------------------------------------------------------
    def close_month(self, period: str) -> PeriodResult:
        started = dt.datetime.now()
        t = period_to_idx(period)
        try:
            sd = self.source(month_end(t))
            if sd.errors:
                msg = "; ".join(f"{e['check']}: {e['message']} ({e['count']})" for e in sd.errors)
                raise DataQualityError("Ошибки качества данных: " + msg)
            self._check_sequence(period)
            c, f, arrays, curves = self.prepared()
            prev = self.store.group_state(idx_to_period(t - 1))
            cg = self.store.contract_groups()
            res = close_period(t, c, f, self.cfg, curves, prev, cg, arrays)
            self.store.save_period(res)
            failed = int((~res.checks.loc[res.checks["severity"] == "control", "passed"]).sum()) if len(res.checks) else 0
            self.store.log_run(period, started, "OK" if failed == 0 else "CHECKS_FAILED",
                               n_contracts=res.info["n_existing"] + res.info["n_new"], n_new=res.info["n_new"],
                               checks_failed=failed,
                               assumptions=f"{res.info['assumptions_old']}→{res.info['assumptions_new']}",
                               config_hash=self.cfg.fingerprint(),
                               message="; ".join(res.info.get("curve_warnings", [])))
            return res
        except Exception as e:
            self.store.log_run(period, started, "ERROR", config_hash=self.cfg.fingerprint(), message=str(e))
            raise

    def close_range(self, p_from: str, p_to: str, excel: bool = True) -> RunSummary:
        t0 = time.perf_counter()
        summary = RunSummary()
        for p in period_range(p_from, p_to):
            r = self.close_month(p)
            summary.results.append(r)
            log.info("%s: договоров %d (новых %d), контроли %s", p, r.info["n_existing"] + r.info["n_new"],
                     r.info["n_new"], "OK" if r.all_passed else "НЕ ПРОЙДЕНЫ")
        summary.dq_issues = self.source().issues
        if excel and summary.results:
            summary.excel_path = str(self.write_excel(p_to))
        summary.seconds = time.perf_counter() - t0
        return summary

    def write_excel(self, period: str):
        from .excel_report import write_report

        _, _, _, curves = self.prepared()
        out = self.cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"IFRS17_annuity_{period}.xlsx"
        write_report(self.cfg, self.store, period, path, curves=curves, dq_issues=self.source().issues)
        return path
