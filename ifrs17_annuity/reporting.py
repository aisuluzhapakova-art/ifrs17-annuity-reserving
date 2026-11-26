"""Отчетные формы: роллфорварды п.100/101, выручка п.106, баланс, ОПиУ, анализ изменений; YTD-агрегация.

Знаки: в роллфорвардах обязательства «+»; в ОПиУ доход «+», расход «−».
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .dates import idx_to_period, period_to_idx, year_start_idx
from .store import ResultStore

P100_COLS = [("LRC_EXCL", "ООП без компонента убытка"), ("LC", "Компонент убытка"), ("LIC", "ОВУ")]
P100_LINES = [
    ("OPEN", "Входящий остаток"),
    ("REV", "Выручка по страхованию"),
    ("ISE_CLAIMS", "Понесенные убытки и прочие расходы по страховым услугам"),
    ("ISE_IACF", "Амортизация аквизиционных денежных потоков"),
    ("ISE_LOSS", "Убытки и восстановление убытков по обременительным группам"),
    ("ISE_LCALLOC", "Распределение компонента убытка (п.50(a))"),
    ("IC", "Инвестиционный компонент"),
    ("IFIE", "Финансовые расходы по страхованию"),
    ("CASH_PREM", "Полученные премии"),
    ("CASH_IACF", "Уплаченные аквизиционные денежные потоки"),
    ("CASH_CLAIMS", "Оплаченные выплаты и расходы"),
    ("CLOSE", "Исходящий остаток"),
]
P101_COLS = [("PVFCF", "PV будущих денежных потоков (вкл. ОВУ)"), ("RA", "Рисковая поправка"), ("CSM", "CSM")]
P101_LINES = [
    ("OPEN", "Входящий остаток"),
    ("CS_CSM", "Текущая услуга: высвобождение CSM"),
    ("CS_RA", "Текущая услуга: изменение RA"),
    ("CS_EXP", "Текущая услуга: корректировки на основе опыта"),
    ("FS_EST", "Будущая услуга: изменения оценок, корректирующие CSM"),
    ("FS_NEW", "Будущая услуга: договоры, признанные в периоде"),
    ("PS_LIC", "Прошлая услуга: изменения ОВУ"),
    ("IFIE", "Финансовые расходы по страхованию"),
    ("CASH", "Денежные потоки"),
    ("CLOSE", "Исходящий остаток"),
]
PL_LINES = [
    ("REV_CLAIMS", "  Ожидаемые выплаты (без ИК)"),
    ("REV_EXP", "  Ожидаемые расходы"),
    ("REV_RA", "  Изменение рисковой поправки"),
    ("REV_CSM", "  Высвобождение CSM"),
    ("REV_IACF", "  Возмещение аквизиционных денежных потоков"),
    ("REV_TOTAL", "Выручка по страхованию"),
    ("ISE_CLAIMS", "  Понесенные выплаты (без ИК)"),
    ("ISE_EXP", "  Понесенные расходы"),
    ("ISE_IACF", "  Амортизация аквизиционных денежных потоков"),
    ("ISE_LOSS", "  Убытки / восстановление по обременительным группам"),
    ("ISE_LCALLOC", "  Распределение компонента убытка"),
    ("ISE_TOTAL", "Расходы по страховым услугам"),
    ("ISR", "Результат страховых услуг"),
    ("IFIE_PL", "Финансовые доходы/расходы по страхованию в ОПиУ"),
    ("PROFIT", "Прибыль/убыток"),
    ("IFIE_OCI", "Финансовые доходы/расходы по страхованию в ПСД"),
    ("TCI", "Совокупный доход"),
    ("MEMO_EXPER", "Справочно: отклонение ожидаемых потоков от фактических (ожид. − факт.)"),
    ("MEMO_IC_ACT", "Справочно: инвестиционный компонент фактический"),
    ("MEMO_IC_EXP", "Справочно: инвестиционный компонент ожидаемый"),
]
REV106_LINES = [l for l in PL_LINES if l[0].startswith("REV_")]
BS_LINES = [
    ("LRC_EXCL", "ООП без компонента убытка"), ("LC", "Компонент убытка"), ("LIC", "ОВУ"),
    ("TOTAL", "Итого обязательства по договорам страхования"),
    ("BEL", "  в т.ч. BEL (ООП)"), ("RA", "  в т.ч. рисковая поправка"), ("CSM", "  в т.ч. CSM"),
    ("BEL_LOCKED", "Справочно: BEL по зафиксированным ставкам"),
    ("OCI_ACC", "Справочно: накопленный резерв ПСД (BEL тек. − BEL зафикс.)"),
    ("IACF", "Справочно: неамортизированные аквизиционные потоки"),
]
AOC_LINES = [
    ("S0", "0. Входящий остаток"),
    ("S0A", "0a. Пересчет по текущим данным DWH"),
    ("S1", "1. Процентный рост"),
    ("S2", "2. Ожидаемые выплаты и расходы месяца / высвобождение RA"),
    ("S3", "3. Фактический состав портфеля"),
    ("S4", "4. Новые нефинансовые допущения"),
    ("S5", "5. Новая кривая на конец месяца"),
    ("S6", "6. Новые договоры"),
    ("SC", "Исходящий остаток"),
]
CSM_LINES = [("OPEN", "Входящий CSM"), ("INT", "Проценты по зафиксированной ставке"),
             ("FS", "Изменения, относящиеся к будущей услуге"), ("NEW", "Новые договоры"),
             ("REL", "Высвобождение CSM"), ("CLOSE", "Исходящий CSM")]
LC_LINES = [("OPEN", "Входящий LC"), ("FS", "Неблагоприятные изменения / восстановление"),
            ("NEW", "Обременительные новые договоры"), ("ALLOC_S", "Распределение: услуги"),
            ("ALLOC_F", "Распределение: финансовая часть"), ("CLOSE", "Исходящий LC")]
STOCK_LINES = {"OPEN", "CLOSE", "S0", "SC"}


def portfolio_of_group(group_id: str) -> str:
    return str(group_id).split("|")[0]


@dataclass
class ReportData:
    period: str
    lines: pd.DataFrame      # report_lines с начала года до периода (+ помесячная динамика)
    units: float


class Reporter:
    def __init__(self, cfg: Config, store: ResultStore):
        self.cfg = cfg
        self.store = store
        self.units = cfg.units

    def load(self, period: str, portfolios: list[str] | None = None, groups: list[str] | None = None) -> ReportData:
        t = period_to_idx(period)
        df = self.store.report_lines(idx_to_period(year_start_idx(t)), period)
        if portfolios:
            df = df[df["group_id"].map(portfolio_of_group).isin(portfolios)]
        if groups:
            df = df[df["group_id"].isin(groups)]
        return ReportData(period=period, lines=df, units=self.units)

    # --- агрегирование -----------------------------------------------------
    @staticmethod
    def _agg(df: pd.DataFrame, report: str, period: str, ytd: bool, by_group: bool = False) -> pd.DataFrame:
        d = df[df["report"] == report]
        if d.empty:
            return pd.DataFrame(columns=["group_id", "line", "col", "value"]) if by_group else pd.DataFrame(columns=["line", "col", "value"])
        keys = (["group_id"] if by_group else []) + ["line", "col"]
        if not ytd:
            d = d[d["period"] == period]
            return d.groupby(keys, as_index=False)["value"].sum()
        first = d["period"].min()
        flows = d[~d["line"].isin(STOCK_LINES)]
        opens = d[(d["line"].isin({"OPEN", "S0"})) & (d["period"] == first)]
        closes = d[(d["line"].isin({"CLOSE", "SC"})) & (d["period"] == period)]
        if report in ("PL", "MEMO"):
            opens = closes = d.iloc[0:0]
        return pd.concat([opens, flows, closes]).groupby(keys, as_index=False)["value"].sum()

    def _table(self, agg: pd.DataFrame, lines, cols, total: bool = True) -> pd.DataFrame:
        piv = agg.pivot_table(index="line", columns="col", values="value", aggfunc="sum") if len(agg) else pd.DataFrame()
        out = pd.DataFrame(index=[l for l, _ in lines])
        for c, _ in cols:
            out[c] = piv[c] if c in piv.columns else 0.0
        out = out.reindex([l for l, _ in lines]).fillna(0.0) / self.units
        out.columns = [lbl for _, lbl in cols]
        if total and len(cols) > 1:
            out["Итого"] = out.sum(axis=1)
        out.index = [lbl for _, lbl in lines]
        out.index.name = "Статья"
        return out

    def rollforward(self, rd: ReportData, which: str, ytd: bool = False, group: str | None = None) -> pd.DataFrame:
        df = rd.lines if group is None else rd.lines[rd.lines["group_id"] == group]
        if which == "P100":
            return self._table(self._agg(df, "P100", rd.period, ytd), P100_LINES, P100_COLS)
        return self._table(self._agg(df, "P101", rd.period, ytd), P101_LINES, P101_COLS)

    def pl(self, rd: ReportData) -> pd.DataFrame:
        m = self._table(self._agg(rd.lines, "PL", rd.period, False), PL_LINES, [("AMT", "За месяц")], total=False)
        y = self._table(self._agg(rd.lines, "PL", rd.period, True), PL_LINES, [("AMT", "С начала года")], total=False)
        return pd.concat([m, y], axis=1)

    def pl_dynamics(self, rd: ReportData) -> pd.DataFrame:
        d = rd.lines[rd.lines["report"] == "PL"]
        piv = d.pivot_table(index="line", columns="period", values="value", aggfunc="sum") / self.units
        piv = piv.reindex([l for l, _ in PL_LINES]).fillna(0.0)
        piv.index = [lbl for _, lbl in PL_LINES]
        piv.index.name = "Статья"
        return piv

    def revenue(self, rd: ReportData) -> pd.DataFrame:
        m = self._table(self._agg(rd.lines, "PL", rd.period, False), REV106_LINES + [("REV_TOTAL", "Итого выручка")],
                        [("AMT", "За месяц")], total=False)
        y = self._table(self._agg(rd.lines, "PL", rd.period, True), REV106_LINES + [("REV_TOTAL", "Итого выручка")],
                        [("AMT", "С начала года")], total=False)
        return pd.concat([m, y], axis=1)

    def balance(self, rd: ReportData) -> pd.DataFrame:
        d = rd.lines[rd.lines["report"] == "BS"]
        first = d["period"].min() if len(d) else rd.period
        cols = {
            "На начало года": d[(d["period"] == first) & (d["col"] == "OPEN")],
            "На начало месяца": d[(d["period"] == rd.period) & (d["col"] == "OPEN")],
            "На конец месяца": d[(d["period"] == rd.period) & (d["col"] == "CLOSE")],
        }
        out = pd.DataFrame(index=[l for l, _ in BS_LINES])
        for name, x in cols.items():
            out[name] = x.groupby("line")["value"].sum()
        out = out.fillna(0.0) / self.units
        out.index = [lbl for _, lbl in BS_LINES]
        out.index.name = "Статья"
        return out

    def balance_by_portfolio(self, rd: ReportData) -> pd.DataFrame:
        d = rd.lines[(rd.lines["report"] == "BS") & (rd.lines["period"] == rd.period) & (rd.lines["col"] == "CLOSE")].copy()
        d["portfolio"] = d["group_id"].map(portfolio_of_group)
        piv = d.pivot_table(index="line", columns="portfolio", values="value", aggfunc="sum") / self.units
        piv = piv.reindex([l for l, _ in BS_LINES]).fillna(0.0)
        piv.index = [lbl for _, lbl in BS_LINES]
        piv.index.name = "Статья (на конец месяца)"
        return piv

    def aoc(self, rd: ReportData, group: str | None = None) -> pd.DataFrame:
        df = rd.lines if group is None else rd.lines[rd.lines["group_id"] == group]
        main = self._table(self._agg(df, "AOC", rd.period, False), AOC_LINES,
                           [("BEL", "BEL (текущие ставки)"), ("BEL_LOCKED", "BEL (зафиксированные ставки)"),
                            ("RA", "RA")], total=False)
        main["Разница тек. − зафикс. (шаг 4a → IFIE)"] = main.iloc[:, 0] - main.iloc[:, 1]
        return main

    def csm_lc_steps(self, rd: ReportData, group: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = rd.lines if group is None else rd.lines[rd.lines["group_id"] == group]
        c = self._table(self._agg(df, "AOC_CSM", rd.period, False), CSM_LINES, [("CSM", "CSM")], total=False)
        lc = self._table(self._agg(df, "AOC_LC", rd.period, False), LC_LINES, [("LC", "LC")], total=False)
        return c, lc

    def memo(self, rd: ReportData) -> dict:
        d = rd.lines[(rd.lines["report"] == "MEMO") & (rd.lines["period"] == rd.period)]
        return d.groupby("line")["value"].sum().to_dict()

    def groups(self, rd: ReportData) -> pd.DataFrame:
        gs = self.store.group_state(rd.period)
        if gs.empty:
            return pd.DataFrame()
        cg = self.store.contract_groups()
        groups = sorted(rd.lines["group_id"].unique())
        gs = gs[gs["group_id"].isin(groups)].set_index("group_id")
        d = rd.lines[(rd.lines["period"] == rd.period) & (rd.lines["report"].isin(["PL", "MEMO"]))]
        pl = d.pivot_table(index="group_id", columns="line", values="value", aggfunc="sum")
        comp = cg.groupby("group_id").agg(n=("contract_id", "count"), premium=("premium", "sum"),
                                          margin=("csm_init", "sum"), first=("recog_period", "min"),
                                          last=("recog_period", "max"))
        u = self.units
        out = pd.DataFrame(index=gs.index)
        out["Портфель"] = [portfolio_of_group(g) for g in gs.index]
        out["Когорта"] = [g.split("|")[1] if "|" in g else "" for g in gs.index]
        out["Прибыльность"] = [g.split("|")[2] if g.count("|") >= 2 else "" for g in gs.index]
        out["Договоров"] = comp["n"].reindex(gs.index).fillna(0).astype(int)
        out["Действующих"] = pl.get("N_INFORCE", pd.Series(dtype=float)).reindex(gs.index).fillna(0).astype(int)
        out["Премия при признании"] = comp["premium"].reindex(gs.index).fillna(0) / u
        out["Маржа при признании (CSM−LC)"] = comp["margin"].reindex(gs.index).fillna(0) / u
        out["Маржа, % премии"] = (comp["margin"] / comp["premium"]).reindex(gs.index)
        out["BEL"] = gs["bel"] / u
        out["RA"] = gs["ra"] / u
        out["CSM"] = gs["csm"] / u
        out["LC"] = gs["lc"] / u
        out["ОВУ"] = gs["lic"] / u
        out["Итого обязательства"] = (gs["bel"] + gs["ra"] + gs["csm"] + gs["lic"]) / u
        for code, lbl in [("REV_TOTAL", "Выручка за месяц"), ("ISR", "Результат услуг за месяц"),
                          ("REV_CSM", "Высвобождение CSM за месяц"), ("IFIE_PL", "IFIE в ОПиУ за месяц")]:
            out[lbl] = pl.get(code, pd.Series(dtype=float)).reindex(gs.index).fillna(0) / u
        out.index.name = "Группа"
        return out

    def group_composition(self) -> pd.DataFrame:
        cg = self.store.contract_groups()
        if cg.empty:
            return cg
        out = cg[["contract_id", "group_id", "portfolio", "cohort", "profitability", "recog_period", "premium",
                  "acquisition_cost", "bel_init", "ra_init", "csm_init", "margin_pct"]].copy()
        out.columns = ["Договор", "Группа", "Портфель", "Когорта", "Прибыльность", "Месяц признания", "Премия",
                       "IACF", "BEL при признании", "RA при признании", "CSM при признании (до LC)", "Маржа, %"]
        return out.sort_values(["Группа", "Договор"])

    def checks(self, period: str) -> pd.DataFrame:
        c = self.store.checks(period)
        if c.empty:
            return c
        c = c[["group_id", "check", "diff", "passed", "severity"]].copy()
        c.columns = ["Группа", "Контроль", "Расхождение, KZT", "Пройден", "Тип"]
        return c

    def summary(self, rd: ReportData) -> pd.DataFrame:
        bs = self.balance(rd)
        pl = self.pl(rd)
        chk = self.store.checks(rd.period)
        ctrl = chk[chk["severity"] == "control"] if len(chk) else chk
        memo = self.memo(rd)
        rows = [
            ("Отчетная дата", rd.period, ""),
            ("Итого обязательства", bs.loc["Итого обязательства по договорам страхования", "На конец месяца"],
             bs.loc["Итого обязательства по договорам страхования", "На начало года"]),
            ("CSM", bs.loc["  в т.ч. CSM", "На конец месяца"], bs.loc["  в т.ч. CSM", "На начало года"]),
            ("Компонент убытка", bs.loc["Компонент убытка", "На конец месяца"], bs.loc["Компонент убытка", "На начало года"]),
            ("Рисковая поправка", bs.loc["  в т.ч. рисковая поправка", "На конец месяца"], bs.loc["  в т.ч. рисковая поправка", "На начало года"]),
            ("BEL", bs.loc["  в т.ч. BEL (ООП)", "На конец месяца"], bs.loc["  в т.ч. BEL (ООП)", "На начало года"]),
            ("ОВУ", bs.loc["ОВУ", "На конец месяца"], bs.loc["ОВУ", "На начало года"]),
            ("Выручка по страхованию (месяц / YTD)", pl.loc["Выручка по страхованию", "За месяц"], pl.loc["Выручка по страхованию", "С начала года"]),
            ("Результат страховых услуг (месяц / YTD)", pl.loc["Результат страховых услуг", "За месяц"], pl.loc["Результат страховых услуг", "С начала года"]),
            ("IFIE в ОПиУ (месяц / YTD)", pl.loc["Финансовые доходы/расходы по страхованию в ОПиУ", "За месяц"], pl.loc["Финансовые доходы/расходы по страхованию в ОПиУ", "С начала года"]),
            ("IFIE в ПСД (месяц / YTD)", pl.loc["Финансовые доходы/расходы по страхованию в ПСД", "За месяц"], pl.loc["Финансовые доходы/расходы по страхованию в ПСД", "С начала года"]),
            ("Действующих договоров", memo.get("N_INFORCE", 0), ""),
            ("Новых договоров за месяц", memo.get("N_NEW", 0), ""),
            ("Групп", int(rd.lines.loc[rd.lines["period"] == rd.period, "group_id"].nunique()), ""),
            ("Контроли пройдены", f"{int(ctrl['passed'].sum())} из {len(ctrl)}" if len(ctrl) else "нет данных", ""),
        ]
        return pd.DataFrame(rows, columns=["Показатель", "Значение", "На начало года / YTD"]).set_index("Показатель")

    def parameters(self) -> pd.DataFrame:
        p = self.cfg.policy
        d = self.cfg.discount
        rows = [
            ("Организация", self.cfg.entity.get("name", "")), ("Валюта", self.cfg.entity.get("currency", "KZT")),
            ("Единицы отчетов", f"{self.units:,.0f} {self.cfg.entity.get('currency', 'KZT')}"),
            ("Модель оценки", p.get("measurement_model")), ("Дезагрегация IFIE", self.cfg.ifie_policy),
            ("Дезагрегация RA", "да" if self.cfg.ra_disaggregation else "нет"),
            ("Промежуточная отчетность (B137)", p.get("interim_approach")),
            ("Единицы покрытия", p.get("coverage_units")),
            ("Гарантированный период как ИК", "да" if self.cfg.guarantee_as_ic else "нет"),
            ("Порог «без значительного риска»", p.get("no_sig_risk_threshold")),
            ("Премия за неликвидность", d.get("illiquidity_premium")), ("LLP, лет", d.get("llp_years")),
            ("UFR", d.get("ufr")), ("Период сходимости, лет", d.get("convergence_years")),
            ("Допуск контролей, KZT", self.cfg.tolerance), ("Хэш конфигурации", self.cfg.fingerprint()),
        ]
        return pd.DataFrame(rows, columns=["Параметр", "Значение"]).set_index("Параметр")
