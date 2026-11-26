"""Excel-пакет отчетности."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import Config
from .reporting import Reporter
from .store import ResultStore

NUM_FMT = '#,##0;[Red]-#,##0;"–"'
HDR_FILL = PatternFill("solid", fgColor="DDE5F0")
BOLD_LINES = ("Входящий", "Исходящий", "Итого", "Выручка по страхованию", "Расходы по страховым услугам",
              "Результат страховых услуг", "Прибыль", "Совокупный", "Входящий CSM", "Исходящий CSM")


class _Sheet:
    def __init__(self, writer, name: str, title: str):
        self.writer, self.name, self.row = writer, name, 0
        pd.DataFrame([[title]]).to_excel(writer, sheet_name=name, header=False, index=False, startrow=0)
        self.ws = writer.sheets[name]
        self.ws.cell(1, 1).font = Font(bold=True, size=13)
        self.row = 2

    def note(self, text: str):
        self.ws.cell(self.row + 1, 1, text).font = Font(italic=True, color="555555")
        self.row += 1

    def table(self, df: pd.DataFrame, caption: str | None = None, index: bool = True, pct_cols=()):
        if caption:
            self.ws.cell(self.row + 2, 1, caption).font = Font(bold=True)
            self.row += 2
        start = self.row + 1
        df.to_excel(self.writer, sheet_name=self.name, startrow=start - 1 + 1, index=index)
        hdr = start + 1
        ncols = df.shape[1] + (1 if index else 0)
        for c in range(1, ncols + 1):
            cell = self.ws.cell(hdr, c)
            cell.fill, cell.font = HDR_FILL, Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for r in range(hdr + 1, hdr + 1 + len(df)):
            first = str(self.ws.cell(r, 1).value or "")
            bold = first.strip().startswith(BOLD_LINES)
            for c in range(1, ncols + 1):
                cell = self.ws.cell(r, c)
                if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                    hdr_name = str(self.ws.cell(hdr, c).value or "")
                    cell.number_format = "0.0%" if hdr_name in pct_cols else NUM_FMT
                if bold:
                    cell.font = Font(bold=True)
        self.row = hdr + len(df) + 1

    def widths(self, first: int = 60, other: int = 18):
        self.ws.column_dimensions["A"].width = first
        for c in range(2, self.ws.max_column + 1):
            self.ws.column_dimensions[get_column_letter(c)].width = other
        self.ws.freeze_panes = "B4"


def write_report(cfg: Config, store: ResultStore, period: str, path: str | Path, curves=None,
                 dq_issues=None, portfolios=None) -> Path:
    rep = Reporter(cfg, store)
    rd = rep.load(period, portfolios=portfolios)
    if rd.lines.empty:
        raise ValueError(f"Нет результатов за {period}")
    unit = f"{cfg.units:,.0f} {cfg.entity.get('currency', 'KZT')}".replace(",", " ")
    groups = sorted(rd.lines.loc[rd.lines["period"] == period, "group_id"].unique())
    path = Path(path)

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        s = _Sheet(w, "Сводка", f"МСФО 17 — пенсионные аннуитеты. Сводка на конец {period}")
        s.note(f"{cfg.entity.get('name', '')}. Единицы: {unit}")
        s.table(rep.summary(rd), "Ключевые показатели")
        s.table(rep.parameters(), "Параметры расчета")
        log = store.run_log()
        if len(log):
            s.table(log.tail(15).set_index("run_id"), "Журнал запусков (последние)")
        s.widths(45, 22)

        s = _Sheet(w, "Баланс", f"Обязательства по договорам страхования, {period}")
        s.note(f"Единицы: {unit}. Обязательства «+».")
        s.table(rep.balance(rd))
        s.table(rep.balance_by_portfolio(rd), "По портфелям")
        s.widths()

        s = _Sheet(w, "ОПиУ", f"ОПиУ и ПСД, {period}")
        s.note(f"Единицы: {unit}. Доход «+», расход «−». Финансовые расходы показаны без инвестиционного дохода по активам.")
        s.table(rep.pl(rd))
        s.table(rep.pl_dynamics(rd), "Помесячная динамика")
        s.widths()

        for which, name, title in [("P100", "П100", "Сверка п.100: ООП / компонент убытка / ОВУ"),
                                   ("P101", "П101", "Сверка п.101: PV ДП / RA / CSM")]:
            s = _Sheet(w, name, f"{title}, {period}")
            s.note(f"Единицы: {unit}. Обязательства «+».")
            s.table(rep.rollforward(rd, which, ytd=False), "За месяц — итого")
            s.table(rep.rollforward(rd, which, ytd=True), "С начала года — итого")
            for g in groups:
                s.table(rep.rollforward(rd, which, ytd=False, group=g), f"За месяц — группа {g}")
            s.widths()

        s = _Sheet(w, "Выручка п.106", f"Анализ выручки (п.106), {period}")
        s.note(f"Единицы: {unit}")
        s.table(rep.revenue(rd))
        s.widths()

        s = _Sheet(w, "Анализ изменений", f"Анализ изменений BEL / RA / CSM / LC, {period}")
        s.note(f"Единицы: {unit}. Шаг 4a — разница шагов 0a, 3, 4 по текущим и зафиксированным ставкам; относится в IFIE.")
        s.table(rep.aoc(rd), "Итого")
        c, lc = rep.csm_lc_steps(rd)
        s.table(c, "CSM")
        s.table(lc, "Компонент убытка")
        for g in groups:
            s.table(rep.aoc(rd, g), f"Группа {g}")
        s.widths()

        s = _Sheet(w, "Группы", f"Группы договоров на конец {period}")
        s.note(f"Единицы: {unit}")
        s.table(rep.groups(rd), pct_cols=("Маржа, % премии",))
        s.widths(40, 16)

        comp = rep.group_composition()
        if len(comp):
            s = _Sheet(w, "Состав групп", "Отнесение договоров к группам и маржа при признании (KZT)")
            s.table(comp, index=False, pct_cols=("Маржа, %",))
            s.widths(16, 16)

        s = _Sheet(w, "Контроли", f"Контроли целостности, {period}. Допуск {cfg.tolerance} KZT")
        s.table(rep.checks(period), index=False)
        if dq_issues:
            s.table(pd.DataFrame(dq_issues), "Проверки качества данных", index=False)
        s.widths(30, 22)

        s = _Sheet(w, "Допущения", f"Допущения и кривая на {period}")
        from .dates import period_to_idx
        t = period_to_idx(period)
        a = cfg.assumptions_at(t)
        s.table(pd.DataFrame(list(a.as_dict().items()), columns=["Параметр", "Значение"]).set_index("Параметр"),
                "Действующая версия допущений")
        allv = pd.DataFrame([x.as_dict() for x in cfg.assumptions]).set_index("Версия").T
        s.table(allv, "Все версии допущений")
        if curves is not None:
            st = curves.spot_table(t)
            s.table(st.set_index("Срок, лет"), "Кривая дисконтирования (текущая)")
            for col in range(2, 6):
                for r in range(s.row - len(st) + 1, s.row + 1):
                    s.ws.cell(r, col).number_format = "0.000%" if col < 5 else "0.000000"
        s.widths(45, 18)
    return path
