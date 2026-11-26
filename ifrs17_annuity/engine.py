"""Ежемесячный расчет по МСФО 17 (GMM): анализ изменений, CSM, LC, LIC, IFIE/ПСД, контроли.

Обозначения (суммы по группе, KZT):
  C*  — BEL по текущим ставкам, L* — BEL по зафиксированным ставкам, RA* — рисковая поправка.
  0   — входящий остаток (исходящий прошлого месяца);
  0a  — пересчет на начало месяца по текущим данным DWH;
  2   — после процентов и ожидаемых потоков месяца (статус на начало месяца);
  3   — фактический состав портфеля;
  4   — новые нефинансовые допущения;
  5   — новая кривая на конец месяца (только текущие ставки).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .curves import CurveSet
from .dates import idx_to_period, series_to_idx
from .projection import PortfolioArrays, horizon, project, pv

STATE_COLS = ["bel", "bel_locked", "ra", "csm", "lc", "lic", "iacf_unamort", "oci_accum",
              "cum_due_nonic", "cum_due_ic", "cum_due_exp", "cum_paid"]


@dataclass
class PeriodResult:
    period: str
    group_state: pd.DataFrame
    new_contract_groups: pd.DataFrame
    group_detail: pd.DataFrame
    report_lines: pd.DataFrame
    checks: pd.DataFrame
    info: dict = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        c = self.checks
        return bool(c.loc[c["severity"] == "control", "passed"].all()) if len(c) else True


def _gsum(values: np.ndarray, groups: np.ndarray, keys: list[str]) -> dict[str, float]:
    s = pd.Series(values, dtype=float).groupby(pd.Series(groups)).sum()
    return {g: float(s.get(g, 0.0)) for g in keys}


def _profitability(margin_amt: np.ndarray, premium: np.ndarray, threshold: float) -> np.ndarray:
    ratio = np.divide(margin_amt, premium, out=np.zeros_like(margin_amt), where=premium > 0)
    return np.where(margin_amt < 0, "ONEROUS", np.where(ratio >= threshold, "NO_SIG_RISK", "OTHER"))


def actual_flows(cf: pd.DataFrame, contracts: pd.DataFrame, cg_map: pd.Series, t: int,
                 guarantee_as_ic: bool) -> pd.DataFrame:
    """Накопленные по группам суммы начисленных и оплаченных потоков на конец месяца t."""
    if cf.empty:
        return pd.DataFrame(columns=["cum_due_nonic", "cum_due_ic", "cum_due_exp", "cum_paid"])
    f = cf[cf["cf_type"].isin(["ANNUITY", "EXPENSE", "SURRENDER"])]
    f = f[f["due_idx"] <= t]
    f = f[f["contract_id"].isin(cg_map.index)]
    if f.empty:
        return pd.DataFrame(columns=["cum_due_nonic", "cum_due_ic", "cum_due_exp", "cum_paid"])
    g_end = contracts.set_index("contract_id")["g_end_idx"]
    ge = f["contract_id"].map(g_end).to_numpy()
    is_ann = (f["cf_type"] == "ANNUITY").to_numpy()
    in_g = f["due_idx"].to_numpy() <= ge
    amt = f["amount"].to_numpy(float)
    ic = (f["cf_type"] == "SURRENDER").to_numpy() | (is_ann & in_g & guarantee_as_ic)
    nonic = is_ann & ~ic
    expn = (f["cf_type"] == "EXPENSE").to_numpy()
    paid = (f["paid_idx"].to_numpy() <= t)
    df = pd.DataFrame({
        "group_id": f["contract_id"].map(cg_map).to_numpy(),
        "cum_due_nonic": amt * nonic,
        "cum_due_ic": amt * ic,
        "cum_due_exp": amt * expn,
        "cum_paid": amt * paid,
    })
    return df.groupby("group_id").sum()


def prepare_contracts(contracts: pd.DataFrame, cashflows: pd.DataFrame, cfg: Config):
    """Добавляет индексы месяцев, используемые движком."""
    c = contracts.copy()
    c["contract_id"] = c["contract_id"].astype(str)
    c["conclusion_idx"] = series_to_idx(c["conclusion_date"])
    c["s_idx"] = series_to_idx(c["payment_start_date"])
    c["g_end_idx"] = c["s_idx"] + c["guarantee_months"].fillna(0).astype(int) - 1
    c["portfolio"] = c["product_code"].map(cfg.portfolios)
    c = c.reset_index(drop=True)
    f = cashflows.copy()
    if len(f):
        f["contract_id"] = f["contract_id"].astype(str)
        f["cf_type"] = f["cf_type"].astype(str).str.upper()
        f["due_idx"] = series_to_idx(f["due_date"])
        f["paid_idx"] = series_to_idx(f["paid_date"])
    else:
        f = pd.DataFrame(columns=["contract_id", "cf_type", "due_date", "paid_date", "amount", "due_idx", "paid_idx"])
    return c, f


def close_period(t: int, contracts: pd.DataFrame, cashflows: pd.DataFrame, cfg: Config,
                 curves: CurveSet, prev_state: pd.DataFrame, contract_groups: pd.DataFrame,
                 arrays: PortfolioArrays | None = None) -> PeriodResult:
    """Расчет месяца t. `contracts`/`cashflows` — результат prepare_contracts."""
    period = idx_to_period(t)
    tol = cfg.tolerance
    gic = cfg.guarantee_as_ic
    max_age = cfg.max_age
    A_old, A_new = cfg.assumptions_at(t - 1), cfg.assumptions_at(t)
    same_assumptions = A_old is A_new
    if arrays is None:
        arrays = PortfolioArrays.from_contracts(contracts)

    cg = contract_groups.copy()
    cg["contract_id"] = cg["contract_id"].astype(str)
    cg_idx = cg.set_index("contract_id")
    cid = contracts["contract_id"].to_numpy()

    ex_mask = np.isin(cid, cg_idx.index.to_numpy())
    new_mask = (~ex_mask) & (contracts["conclusion_idx"].to_numpy() <= t) & contracts["portfolio"].notna().to_numpy()

    E = arrays.subset(ex_mask)
    Nw = arrays.subset(new_mask)
    e_cid = cid[ex_mask]
    e_grp = cg_idx.loc[e_cid, "group_id"].to_numpy() if len(e_cid) else np.array([], dtype=object)
    e_tau = (cg_idx.loc[e_cid, "recog_period"].map(lambda p: int(p[:4]) * 12 + int(p[5:7]) - 1).to_numpy()
             if len(e_cid) else np.array([], dtype=np.int64))

    N = max(horizon(E, t - 1, max_age), horizon(Nw, t, max_age), 2)
    proj = lambda pa, T, a, s=False: project(pa, T, a, max_age, N, gic, s)  # noqa: E731

    # ---------------- новые договоры (шаг 6) ----------------
    nc = contracts.loc[new_mask].copy()
    df_cur = curves.forward_from(t, 0, N)          # текущая кривая на конец t
    if len(nc):
        pn, pns = proj(Nw, t, A_new), proj(Nw, t, A_new, True)
        bel_n = pv(pn.outflow, df_cur)
        ra_n = pv(pns.outflow, df_cur) - bel_n
        prem = nc["single_premium"].astype(float).to_numpy()
        acq = nc["acquisition_cost"].fillna(0).astype(float).to_numpy()
        margin = prem - acq - bel_n - ra_n
        prof = _profitability(margin, prem, cfg.no_sig_threshold)
        cohort = str(t // 12)
        n_grp = np.array([f"{p}|{cohort}|{pr}" for p, pr in zip(nc["portfolio"], prof)], dtype=object)
        new_cg = pd.DataFrame({
            "contract_id": nc["contract_id"].to_numpy(), "group_id": n_grp,
            "portfolio": nc["portfolio"].to_numpy(), "cohort": int(cohort), "profitability": prof,
            "recog_period": period, "premium": prem, "acquisition_cost": acq,
            "bel_init": bel_n, "ra_init": ra_n, "csm_init": margin,
            "margin_pct": np.divide(margin, prem, out=np.zeros_like(margin), where=prem > 0),
        })
        cu_n_pv = pv(pn.cu, df_cur)
        cu_n_und = pn.cu.sum(axis=1)
    else:
        n_grp = np.array([], dtype=object)
        new_cg = pd.DataFrame(columns=["contract_id", "group_id", "portfolio", "cohort", "profitability",
                                       "recog_period", "premium", "acquisition_cost", "bel_init",
                                       "ra_init", "csm_init", "margin_pct"])
        bel_n = ra_n = prem = acq = margin = cu_n_pv = cu_n_und = np.zeros(0)

    # ---------------- действующие договоры ----------------
    if len(E):
        df_prev = curves.forward_from(t - 1, 0, N)        # кривая начала месяца
        df_roll = curves.forward_from(t - 1, 1, N)        # она же, прокатанная на конец месяца
        LDa = curves.locked_df(e_tau, t - 1, N)          # зафиксированные, от начала месяца
        LDt = curves.locked_df(e_tau, t, N)              # зафиксированные, от конца месяца

        pa, pas = proj(E, t - 1, A_old), proj(E, t - 1, A_old, True)
        cfa, cfas = pa.outflow, pas.outflow
        C0a = pv(cfa, df_prev)
        C2 = (cfa[:, 1:] @ df_prev[1:]) / df_prev[0]
        S0a = pv(cfas, df_prev)
        S2 = (cfas[:, 1:] @ df_prev[1:]) / df_prev[0]
        L0a = pv(cfa, LDa)
        L2 = np.einsum("ij,ij->i", cfa[:, 1:], LDa[:, 1:]) / LDa[:, 0]
        CF1_n, CF1_ic, CF1_e = pa.ann_nonic[:, 0], pa.ann_ic[:, 0], pa.exp[:, 0]
        CU_t = pa.cu[:, 0]
        r_lock = 1.0 / LDa[:, 0] - 1.0
        r_cur = 1.0 / df_prev[0] - 1.0

        pb, pbs = proj(E, t, A_old), proj(E, t, A_old, True)
        C3 = pv(pb.outflow, df_roll)
        S3 = pv(pbs.outflow, df_roll)
        L3 = pv(pb.outflow, LDt)
        if same_assumptions:
            pc, pcs = pb, pbs
        else:
            pc, pcs = proj(E, t, A_new), proj(E, t, A_new, True)
        C4 = pv(pc.outflow, df_roll)
        S4 = pv(pcs.outflow, df_roll)
        L4 = pv(pc.outflow, LDt)
        C5 = pv(pc.outflow, df_cur)
        S5 = pv(pcs.outflow, df_cur)
        cu_fut_pv = pv(pc.cu, LDt)
        cu_fut_und = pc.cu.sum(axis=1)
        status = np.where(E.cancel_idx <= t, "CANCELLED",
                          np.where(E.death_idx <= t, np.where(E.g_end > t, "DEAD_IN_GUARANTEE", "TERMINATED"),
                                   "ALIVE"))
    else:
        z = np.zeros(0)
        C0a = C2 = S0a = S2 = L0a = L2 = CF1_n = CF1_ic = CF1_e = CU_t = r_lock = z
        C3 = S3 = L3 = C4 = S4 = L4 = C5 = S5 = cu_fut_pv = cu_fut_und = z
        r_cur = 0.0
        status = np.array([], dtype=object)
    RA0a, RA2, RA3, RA4, RA5 = S0a - C0a, S2 - C2, S3 - C3, S4 - C4, S5 - C5

    # ---------------- группы ----------------
    prev = prev_state.set_index("group_id") if len(prev_state) else pd.DataFrame(columns=STATE_COLS)
    keys = sorted(set(prev.index) | set(e_grp) | set(n_grp))
    G = lambda v: _gsum(v, e_grp, keys)  # noqa: E731
    Gn = lambda v: _gsum(v, n_grp, keys)  # noqa: E731

    g = {name: G(v) for name, v in dict(
        C0a=C0a, C2=C2, C3=C3, C4=C4, C5=C5, L0a=L0a, L2=L2, L3=L3, L4=L4,
        RA0a=RA0a, RA2=RA2, RA3=RA3, RA4=RA4, RA5=RA5,
        CF1_n=CF1_n, CF1_ic=CF1_ic, CF1_e=CF1_e, CU_t=CU_t,
        CUf=cu_fut_pv, CUu=cu_fut_und, wr=r_lock * L0a, w=L0a).items()}
    gn = {name: Gn(v) for name, v in dict(
        BELn=bel_n, RAn=ra_n, P=prem, AC=acq, M=margin, CUf=cu_n_pv, CUu=cu_n_und).items()}

    # фактические потоки
    cg_map = pd.concat([cg_idx["group_id"], new_cg.set_index("contract_id")["group_id"]])
    act = actual_flows(cashflows, contracts, cg_map, t, gic)

    states, lines, checks = [], [], []
    is_oci = cfg.ifie_policy == "OCI"
    for k in keys:
        o = {c: float(prev.loc[k, c]) if k in prev.index and c in prev.columns and pd.notna(prev.loc[k, c]) else 0.0
             for c in STATE_COLS}
        C0, L0, RA0, CSM0, LC0, LIC0, IACF0, OCI0 = (o["bel"], o["bel_locked"], o["ra"], o["csm"], o["lc"],
                                                     o["lic"], o["iacf_unamort"], o["oci_accum"])
        v = {n: g[n][k] for n in g}
        w = {n: gn[n][k] for n in gn}
        cf1 = v["CF1_n"] + v["CF1_ic"] + v["CF1_e"]

        # проценты (шаг 1) — по разности прямых оценок, что точно равно BEL0a·(1/DF(1)−1)
        int_C = v["C2"] - v["C0a"] + cf1
        int_L = v["L2"] - v["L0a"] + cf1

        # фактические потоки месяца
        a = act.loc[k] if k in act.index else pd.Series(0.0, index=["cum_due_nonic", "cum_due_ic", "cum_due_exp", "cum_paid"])
        act_n = float(a["cum_due_nonic"]) - o["cum_due_nonic"]
        act_ic = float(a["cum_due_ic"]) - o["cum_due_ic"]
        act_e = float(a["cum_due_exp"]) - o["cum_due_exp"]
        paid = float(a["cum_paid"]) - o["cum_paid"]
        incurred = act_n + act_ic + act_e

        # изменения, относящиеся к будущей услуге (по зафиксированным ставкам)
        fs_L = (v["L0a"] - L0) + (v["L3"] - v["L2"]) + (v["L4"] - v["L3"])
        fs_RA = (v["RA0a"] - RA0) + (v["RA3"] - v["RA2"]) + (v["RA4"] - v["RA3"])
        fs_IC = act_ic - v["CF1_ic"]                     # B96(c)
        fs_total = fs_L + fs_RA + fs_IC
        step4a = (v["C0a"] - C0) + (v["C3"] - v["C2"]) + (v["C4"] - v["C3"]) - fs_L

        # RA: текущая услуга и (опционально) финансовая часть
        ra_cs = (v["RA2"] - v["RA0a"]) + (v["RA5"] - v["RA4"])
        ra_fin = 0.0
        if cfg.ra_disaggregation:
            ra_int = v["RA0a"] * r_cur
            ra_fin = ra_int + (v["RA5"] - v["RA4"])
            ra_cs = (v["RA2"] - v["RA0a"]) - ra_int
        ra_rel = -ra_cs                                   # высвобождение RA в выручку

        # ---- CSM / LC ----
        csm, lc = CSM0, LC0
        r_g = v["wr"] / v["w"] if abs(v["w"]) > 1e-9 else 0.0
        int_csm = csm * r_g
        csm += int_csm
        d = fs_total
        if d > 0:
            absorb = min(csm, d)
            csm -= absorb
            dlc_fs = d - absorb
        else:
            rev = min(lc, -d)
            dlc_fs = -rev
            csm += -d - rev
        lc += dlc_fs
        dcsm_fs = -fs_total + dlc_fs

        nm = w["M"]
        if nm >= 0:
            rev = min(lc, nm)
            dlc_new = -rev
            csm += nm - rev
        else:  # убыточные новые договоры; существующая CSM группы (если есть) поглощает убыток первой
            absorb = min(csm, -nm)
            csm -= absorb
            dlc_new = -nm - absorb
        lc += dlc_new
        dcsm_new = nm + dlc_new

        # систематическое распределение LC (п.50(a))
        base_out = v["C0a"] + v["RA0a"]
        alloc_base_s = v["CF1_n"] + v["CF1_e"] + ra_rel
        alloc_base_f = int_C + ra_fin
        if LC0 > 0 and base_out > 1.0:
            alpha = min(1.0, LC0 / base_out)
            alloc_s = max(alpha * alloc_base_s, 0.0)
            alloc_f = max(alpha * alloc_base_f, 0.0)
        elif LC0 > 0:
            alpha, alloc_s, alloc_f = 1.0, lc, 0.0
        else:
            alpha, alloc_s, alloc_f = 0.0, 0.0, 0.0
        tot = alloc_s + alloc_f
        if tot > lc and tot > 0:
            sc = max(lc, 0.0) / tot
            alloc_s, alloc_f = alloc_s * sc, alloc_f * sc
        lc -= alloc_s + alloc_f
        share = alloc_s / alloc_base_s if abs(alloc_base_s) > 1e-9 else 0.0

        # высвобождение CSM (B119)
        cu_t = v["CU_t"]
        cov = cu_t + v["CUf"] + w["CUf"]
        rel = csm * cu_t / cov if cov > 1e-9 else csm
        csm -= rel

        # IACF: амортизация по единицам покрытия (B125)
        covu = cu_t + v["CUu"] + w["CUu"]
        amort = IACF0 * cu_t / covu if covu > 1e-9 else IACF0
        iacf = IACF0 - amort + w["AC"]

        # ---- выручка, расходы, IFIE ----
        rev_claims = (1 - share) * v["CF1_n"]
        rev_exp = (1 - share) * v["CF1_e"]
        rev_ra = (1 - share) * ra_rel
        revenue = rev_claims + rev_exp + rev_ra + rel + amort
        losses = dlc_fs + dlc_new
        ise = act_n + act_e + amort + losses - alloc_s
        isr = revenue - ise

        ifie_bel = int_C + step4a + (v["C5"] - v["C4"])
        ifie_total = ifie_bel + ra_fin + int_csm
        ifie_pl = (int_L + int_csm + ra_fin) if is_oci else ifie_total
        ifie_oci = ifie_total - ifie_pl

        # ---- закрытие ----
        BEL1 = v["C5"] + w["BELn"]
        BELL1 = v["L4"] + w["BELn"]
        RA1 = v["RA5"] + w["RAn"]
        LIC1 = float(a["cum_due_nonic"] + a["cum_due_ic"] + a["cum_due_exp"] - a["cum_paid"])
        OCI1 = OCI0 + ifie_oci if is_oci else 0.0
        cash = w["P"] - w["AC"] - paid

        liab0 = C0 + RA0 + CSM0 + LIC0
        liab1 = BEL1 + RA1 + csm + LIC1
        lrc_excl0 = C0 + RA0 + CSM0 - LC0
        lrc_excl1 = BEL1 + RA1 + csm - lc

        # роллфорвард п.100
        p100 = {
            "REV": (-revenue, 0.0, 0.0),
            "ISE_CLAIMS": (0.0, 0.0, act_n + act_e),
            "ISE_IACF": (amort, 0.0, 0.0),
            "ISE_LOSS": (0.0, losses, 0.0),
            "ISE_LCALLOC": (0.0, -alloc_s, 0.0),
            "IC": (-act_ic, 0.0, act_ic),
            "IFIE": (ifie_total + alloc_f, -alloc_f, 0.0),
            "CASH_PREM": (w["P"], 0.0, 0.0),
            "CASH_IACF": (-w["AC"], 0.0, 0.0),
            "CASH_CLAIMS": (0.0, 0.0, -paid),
        }
        roll100 = np.array([lrc_excl0, LC0, LIC0]) + np.sum(list(p100.values()), axis=0)

        # роллфорвард п.101
        exper = (act_n + act_e) - (v["CF1_n"] + v["CF1_e"])
        p101 = {
            "CS_CSM": (0.0, 0.0, -rel),
            "CS_RA": (0.0, ra_cs, 0.0),
            "CS_EXP": (exper, 0.0, 0.0),
            "FS_EST": (fs_L + fs_IC, fs_RA, dcsm_fs),
            "FS_NEW": (w["BELn"] - w["P"] + w["AC"], w["RAn"], dcsm_new),
            "IFIE": (ifie_bel, ra_fin, int_csm),
            "CASH": (cash, 0.0, 0.0),
        }
        roll101 = np.array([C0 + LIC0, RA0, CSM0]) + np.sum(list(p101.values()), axis=0)

        L = []
        add = lambda rep, line, col, val: L.append((period, k, rep, line, col, float(val)))  # noqa: E731
        for line, vals in [("OPEN", (lrc_excl0, LC0, LIC0))] + list(p100.items()) + [("CLOSE", (lrc_excl1, lc, LIC1))]:
            for col, val in zip(("LRC_EXCL", "LC", "LIC"), vals):
                add("P100", line, col, val)
        for line, vals in [("OPEN", (C0 + LIC0, RA0, CSM0))] + list(p101.items()) + [("CLOSE", (BEL1 + LIC1, RA1, csm))]:
            for col, val in zip(("PVFCF", "RA", "CSM"), vals):
                add("P101", line, col, val)
        pl = {
            "REV_CLAIMS": rev_claims, "REV_EXP": rev_exp, "REV_RA": rev_ra, "REV_CSM": rel, "REV_IACF": amort,
            "REV_TOTAL": revenue,
            "ISE_CLAIMS": -act_n, "ISE_EXP": -act_e, "ISE_IACF": -amort, "ISE_LOSS": -losses,
            "ISE_LCALLOC": alloc_s, "ISE_TOTAL": -ise, "ISR": isr,
            "IFIE_PL": -ifie_pl, "PROFIT": isr - ifie_pl, "IFIE_OCI": -ifie_oci,
            "TCI": isr - ifie_total,
            "MEMO_EXPER": -exper, "MEMO_IC_ACT": act_ic, "MEMO_IC_EXP": v["CF1_ic"],
        }
        for line, val in pl.items():
            add("PL", line, "AMT", val)
        bs = {"LRC_EXCL": (lrc_excl0, lrc_excl1), "LC": (LC0, lc), "LIC": (LIC0, LIC1),
              "TOTAL": (liab0, liab1), "BEL": (C0, BEL1), "BEL_LOCKED": (L0, BELL1), "RA": (RA0, RA1),
              "CSM": (CSM0, csm), "OCI_ACC": (OCI0, OCI1), "IACF": (IACF0, iacf)}
        for line, (op, cl) in bs.items():
            add("BS", line, "OPEN", op)
            add("BS", line, "CLOSE", cl)
        aoc = [
            ("S0", C0, L0, RA0), ("S0A", v["C0a"] - C0, v["L0a"] - L0, v["RA0a"] - RA0),
            ("S1", int_C, int_L, 0.0), ("S2", -cf1, -cf1, v["RA2"] - v["RA0a"]),
            ("S3", v["C3"] - v["C2"], v["L3"] - v["L2"], v["RA3"] - v["RA2"]),
            ("S4", v["C4"] - v["C3"], v["L4"] - v["L3"], v["RA4"] - v["RA3"]),
            ("S5", v["C5"] - v["C4"], 0.0, v["RA5"] - v["RA4"]),
            ("S6", w["BELn"], w["BELn"], w["RAn"]),
            ("SC", BEL1, BELL1, RA1),
        ]
        for line, cb, lb, rb in aoc:
            add("AOC", line, "BEL", cb)
            add("AOC", line, "BEL_LOCKED", lb)
            add("AOC", line, "RA", rb)
        csm_steps = {"OPEN": CSM0, "INT": int_csm, "FS": dcsm_fs, "NEW": dcsm_new, "REL": -rel, "CLOSE": csm}
        for line, val in csm_steps.items():
            add("AOC_CSM", line, "CSM", val)
        lc_steps = {"OPEN": LC0, "FS": dlc_fs, "NEW": dlc_new, "ALLOC_S": -alloc_s, "ALLOC_F": -alloc_f, "CLOSE": lc}
        for line, val in lc_steps.items():
            add("AOC_LC", line, "LC", val)
        memo = {"S4A": step4a, "S7": exper, "IC_DIFF": fs_IC, "CU_T": cu_t, "CU_FUT_PV": v["CUf"] + w["CUf"],
                "ALPHA": alpha, "R_LOCKED": r_g, "N_NEW": float((n_grp == k).sum()),
                "N_INFORCE": float(((e_grp == k) & (status != "CANCELLED") & (status != "TERMINATED")).sum())}
        for line, val in memo.items():
            add("MEMO", line, "AMT", val)
        lines.extend(L)

        # ---- контроли (раздел 8) ----
        def chk(name, diff, severity="control", info=""):
            checks.append({"period": period, "group_id": k, "check": name, "diff": float(diff),
                           "passed": bool(abs(diff) <= tol) if severity == "control" else True,
                           "severity": severity, "info": info})
        chk("1. Роллфорвард ООП п.100 = BEL+RA+CSM", (roll100[0] + roll100[1]) - (BEL1 + RA1 + csm))
        chk("1b. Роллфорвард п.101 = прямая оценка", (roll101[0] - (BEL1 + LIC1)) + (roll101[1] - RA1) + (roll101[2] - csm))
        chk("2. Роллфорвард ОВУ = остаток", roll100[2] - LIC1)
        chk("3. Δ обязательств = ДП − результат услуг + IFIE", (liab1 - liab0) - (cash - isr + ifie_total))
        if is_oci:
            chk("4. Накопленный ПСД = BEL(тек.) − BEL(зафикс.)", OCI1 - (BEL1 - BELL1))
        chk("5a. CSM ≥ 0", min(csm, 0.0))
        chk("5b. Нет одновременно CSM > 0 и LC > 0", min(csm, lc) if (csm > tol and lc > tol) else 0.0)
        chk("5c. LC ≥ 0", min(lc, 0.0))
        chk("6. Пересчет входящего BEL по данным DWH (инф.)", v["C0a"] - C0, severity="info")

        states.append({"period": period, "group_id": k, "bel": BEL1, "bel_locked": BELL1, "ra": RA1,
                       "csm": csm, "lc": lc, "lic": LIC1, "iacf_unamort": iacf, "oci_accum": OCI1,
                       "cum_due_nonic": float(a["cum_due_nonic"]), "cum_due_ic": float(a["cum_due_ic"]),
                       "cum_due_exp": float(a["cum_due_exp"]), "cum_paid": float(a["cum_paid"])})

    # поконтрактная детализация на конец месяца
    det = [pd.DataFrame({"period": period, "contract_id": e_cid, "group_id": e_grp,
                         "bel": C5, "bel_locked": L4, "ra": RA5, "status": status})]
    if len(nc):
        det.append(pd.DataFrame({"period": period, "contract_id": nc["contract_id"].to_numpy(), "group_id": n_grp,
                                 "bel": bel_n, "bel_locked": bel_n, "ra": ra_n, "status": "NEW"}))
    detail = pd.concat(det, ignore_index=True)

    rl = pd.DataFrame(lines, columns=["period", "group_id", "report", "line", "col", "value"])
    info = {"n_existing": int(len(E)), "n_new": int(len(nc)), "assumptions_old": A_old.version,
            "assumptions_new": A_new.version, "horizon": N, "curve_warnings": list(curves.warnings)}
    return PeriodResult(period=period, group_state=pd.DataFrame(states), new_contract_groups=new_cg,
                        group_detail=detail, report_lines=rl, checks=pd.DataFrame(checks), info=info)
