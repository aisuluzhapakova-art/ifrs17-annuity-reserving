"""Веб-интерфейс (Streamlit): streamlit run app.py"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ifrs17_annuity.config import DEFAULT_CONFIG, load_config
from ifrs17_annuity.excel_report import write_report
from ifrs17_annuity.reporting import Reporter, portfolio_of_group
from ifrs17_annuity.runner import Runner

st.set_page_config(page_title="МСФО 17 — аннуитеты", layout="wide")


@st.cache_resource
def get_runner(config_path: str) -> Runner:
    return Runner(load_config(config_path))


def fmt(df: pd.DataFrame, pct_cols=()):
    num = {c: "{:,.0f}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in pct_cols}
    num.update({c: "{:.1%}" for c in pct_cols if c in df.columns})
    return df.style.format(num, thousands=" ")


st.sidebar.title("МСФО 17 — пенсионные аннуитеты")
config_path = st.sidebar.text_input("Файл конфигурации", str(DEFAULT_CONFIG))
runner = get_runner(config_path)
cfg, store = runner.cfg, runner.store

with st.sidebar.expander("Запуск расчета", expanded=not store.closed_periods()):
    last = store.last_closed_period()
    p_one = st.text_input("Месяц (YYYY-MM)", value=last or "2024-01")
    if st.button("Закрыть месяц"):
        with st.spinner("Расчет..."):
            try:
                s = runner.close_range(p_one, p_one)
                (st.success if s.all_passed else st.error)(
                    f"{p_one}: контроли {'пройдены' if s.all_passed else 'НЕ пройдены'}, {s.seconds:.1f} с")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    c1, c2 = st.columns(2)
    p_from = c1.text_input("С", value="2024-01")
    p_to = c2.text_input("По", value=last or "2026-10")
    if st.button("Рассчитать диапазон"):
        with st.spinner("Расчет диапазона..."):
            try:
                s = runner.close_range(p_from, p_to)
                (st.success if s.all_passed else st.error)(
                    f"{len(s.results)} мес., контроли {'пройдены' if s.all_passed else 'НЕ пройдены'}, {s.seconds:.1f} с")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

periods = store.closed_periods()
if not periods:
    st.info("Нет рассчитанных месяцев. Запустите расчет в боковой панели.")
    st.stop()

period = st.sidebar.selectbox("Отчетный месяц", periods[::-1])
rep = Reporter(cfg, store)
all_lines = rep.load(period)
portfolios = sorted({portfolio_of_group(g) for g in all_lines.lines["group_id"].unique()})
sel = st.sidebar.multiselect("Портфели", portfolios, default=portfolios)
rd = rep.load(period, portfolios=sel or None)
groups = sorted(rd.lines.loc[rd.lines["period"] == period, "group_id"].unique())
unit = f"{cfg.units:,.0f} {cfg.entity.get('currency', 'KZT')}".replace(",", " ")

if st.sidebar.button("Сформировать Excel"):
    with tempfile.TemporaryDirectory() as d:
        p = write_report(cfg, store, period, Path(d) / f"IFRS17_annuity_{period}.xlsx",
                         curves=runner.prepared()[3], dq_issues=runner.source().issues, portfolios=sel or None)
        st.sidebar.download_button("Скачать Excel", p.read_bytes(), file_name=p.name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.title(f"Обязательства по договорам аннуитета на конец {period}")
st.caption(f"{cfg.entity.get('name', '')} · единицы: {unit} · IFIE: {cfg.ifie_policy}")

tabs = st.tabs(["Сводка", "Баланс", "ОПиУ", "Роллфорвард п.100", "Роллфорвард п.101", "Выручка",
                "Анализ изменений", "Группы", "Контроли"])

with tabs[0]:
    bs = rep.balance(rd)
    pl = rep.pl(rd)
    chk = store.checks(period)
    ctrl = chk[chk["severity"] == "control"]
    k = st.columns(5)
    k[0].metric("Итого обязательства", f"{bs.iloc[3, 2]:,.0f}".replace(",", " "))
    k[1].metric("CSM", f"{bs.loc['  в т.ч. CSM', 'На конец месяца']:,.0f}".replace(",", " "))
    k[2].metric("Компонент убытка", f"{bs.loc['Компонент убытка', 'На конец месяца']:,.0f}".replace(",", " "))
    k[3].metric("Результат услуг (мес.)", f"{pl.loc['Результат страховых услуг', 'За месяц']:,.0f}".replace(",", " "))
    k[4].metric("Контроли", f"{int(ctrl['passed'].sum())}/{len(ctrl)}")
    st.dataframe(rep.summary(rd).astype(str), width="stretch")
    dyn = rep.pl_dynamics(rep.load(period, portfolios=sel or None))
    st.subheader("Динамика с начала года")
    rows = [r for r in ["Результат страховых услуг", "  Высвобождение CSM", "Прибыль/убыток"] if r in dyn.index]
    st.line_chart(dyn.loc[rows].T)
    with st.expander("Параметры расчета"):
        st.dataframe(rep.parameters().astype(str), width="stretch")

with tabs[1]:
    st.dataframe(fmt(rep.balance(rd)), width="stretch")
    st.subheader("По портфелям")
    st.dataframe(fmt(rep.balance_by_portfolio(rd)), width="stretch")

with tabs[2]:
    st.dataframe(fmt(rep.pl(rd)), width="stretch", height=740)
    st.subheader("Помесячная динамика")
    st.dataframe(fmt(rep.pl_dynamics(rd)), width="stretch")

for i, which in [(3, "P100"), (4, "P101")]:
    with tabs[i]:
        c1, c2 = st.columns(2)
        mode = c1.radio("Период", ["За месяц", "С начала года"], horizontal=True, key=f"m{which}")
        grp = c2.selectbox("Группа", ["Итого"] + groups, key=f"g{which}")
        df = rep.rollforward(rd, which, ytd=mode != "За месяц", group=None if grp == "Итого" else grp)
        st.dataframe(fmt(df), width="stretch", height=460)

with tabs[5]:
    st.dataframe(fmt(rep.revenue(rd)), width="stretch")

with tabs[6]:
    grp = st.selectbox("Группа", ["Итого"] + groups, key="gaoc")
    g = None if grp == "Итого" else grp
    st.dataframe(fmt(rep.aoc(rd, g)), width="stretch")
    c, lc = rep.csm_lc_steps(rd, g)
    c1, c2 = st.columns(2)
    c1.dataframe(fmt(c), width="stretch")
    c2.dataframe(fmt(lc), width="stretch")

with tabs[7]:
    st.dataframe(fmt(rep.groups(rd), pct_cols=("Маржа, % премии",)), width="stretch")
    with st.expander("Состав групп (поконтрактно)"):
        comp = rep.group_composition()
        st.dataframe(comp[comp["Группа"].isin(groups)], width="stretch", height=400)

with tabs[8]:
    c = rep.checks(period)
    bad = c[(c["Тип"] == "control") & ~c["Пройден"]]
    (st.success if bad.empty else st.error)("Все контроли пройдены" if bad.empty else f"Не пройдено: {len(bad)}")
    st.dataframe(c, width="stretch", height=500)
    issues = runner.source().issues
    if issues:
        st.subheader("Проверки качества данных")
        st.dataframe(pd.DataFrame(issues), width="stretch")
    st.subheader("Журнал запусков")
    st.dataframe(store.run_log().tail(50), width="stretch")
