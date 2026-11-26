"""Генератор синтетических демо-данных в формате витрин DWH.

    python -m demo.generate_demo_data [--contracts 2500] [--from 2024-01] [--to 2026-10] [--seed 42]

Создает:
  config/tables/mortality_demo.csv          — демонстрационная таблица смертности (Гомпертц–Мейкхам)
  data/demo/v_annuity_contracts.csv
  data/demo/v_annuity_cashflows.csv
  data/demo/v_discount_curve.csv
Аннуитет каждого договора рассчитывается от BEL по допущениям первой версии и кривой месяца заключения
со случайной нагрузкой, поэтому в портфеле есть прибыльные и обременительные договоры.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ifrs17_annuity.dates import idx_to_period, month_end, period_to_idx  # noqa: E402

TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]


def write_mortality_table(path: Path, max_age: int = 110) -> Path:
    ages = np.arange(max_age + 1)
    def gm(a, b, c):
        mu = a + b * np.power(c, ages)
        return np.clip(1 - np.exp(-mu), 0, 1)
    q_m = gm(0.0006, 0.000045, 1.103)
    q_f = gm(0.0003, 0.000018, 1.108)
    q_m[-1] = q_f[-1] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"age": ages, "qx_M": q_m.round(8), "qx_F": q_f.round(8)}).to_csv(path, index=False)
    return path


def generate_curves(p_from: str, p_to: str, rng: np.random.Generator) -> pd.DataFrame:
    a, b = period_to_idx(p_from), period_to_idx(p_to)
    rows = []
    short, long_ = 0.145, 0.125
    for i in range(a - 1, b + 1):   # с запасом на месяц до первого
        short = float(np.clip(short + rng.normal(-0.0003, 0.0025), 0.10, 0.18))
        long_ = float(np.clip(long_ + rng.normal(0.0, 0.0015), 0.10, 0.15))
        for t in TENORS:
            w = 1 - np.exp(-t / 3.0)
            rows.append({"curve_date": month_end(i).isoformat(), "tenor_years": t,
                         "zero_rate": round(short + (long_ - short) * w, 6)})
    return pd.DataFrame(rows)


def _price(contracts: pd.DataFrame, cfg, curves) -> np.ndarray:
    """A0 = (P·(1−ac−load) − PV фикс. расходов) / PV аннуитета на 1 тенге (BEL по версии 1)."""
    from ifrs17_annuity.projection import PortfolioArrays, horizon, project, pv

    a = cfg.assumptions[0]
    out = np.zeros(len(contracts))
    conc = contracts["conclusion_date"].map(lambda d: d.year * 12 + d.month - 1).to_numpy()
    for t in np.unique(conc):
        rows = np.where(conc == t)[0]
        sub = contracts.iloc[rows]
        pa = PortfolioArrays.from_contracts(sub.assign(initial_monthly_annuity=1.0, death_date=pd.NaT,
                                                       termination_date=pd.NaT, termination_reason=None))
        N = horizon(pa, int(t), cfg.max_age)
        df = curves.forward_from(int(t), 0, N)
        bel1 = pv(project(pa, int(t), a, cfg.max_age, N, cfg.guarantee_as_ic).outflow, df)
        fixed = pv(project(pa.with_a0(0.0), int(t), a, cfg.max_age, N, cfg.guarantee_as_ic).outflow, df)
        unit = bel1 - fixed
        target = sub["single_premium"].to_numpy() * (1 - sub["_ac"].to_numpy() - sub["_load"].to_numpy())
        out[rows] = (target - fixed) / unit
    return out


def generate(out_dir: Path | str | None = None, n_contracts: int = 2500, p_from: str = "2024-01",
             p_to: str = "2026-10", seed: int = 42, cfg=None, mortality_path: Path | None = None) -> Path:
    from ifrs17_annuity.config import load_config
    from ifrs17_annuity.curves import CurveSet

    rng = np.random.default_rng(seed)
    mortality_path = mortality_path or ROOT / "config" / "tables" / "mortality_demo.csv"
    if not Path(mortality_path).exists():
        write_mortality_table(Path(mortality_path))
    cfg = cfg or load_config()
    out_dir = Path(out_dir) if out_dir else cfg.resolve(cfg.source.get("csv_dir", "data/demo"))
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = generate_curves(p_from, p_to, rng)
    curves = CurveSet(curve.assign(curve_date=pd.to_datetime(curve["curve_date"])), cfg.discount)

    a_idx, b_idx = period_to_idx(p_from), period_to_idx(p_to)
    n = n_contracts
    # распределение заключений с ростом продаж
    months = np.arange(a_idx, b_idx + 1)
    wts = np.linspace(1.0, 1.6, len(months))
    conc_idx = rng.choice(months, size=n, p=wts / wts.sum())
    conc_day = rng.integers(1, 28, size=n)
    conc = pd.to_datetime([f"{i // 12}-{i % 12 + 1:02d}-{d:02d}" for i, d in zip(conc_idx, conc_day)])
    s_idx = conc_idx + 1
    pay_start = pd.to_datetime([month_end(i) for i in s_idx])
    gender = np.where(rng.random(n) < 0.5, "M", "F")
    age_at = np.where(gender == "M", rng.integers(55, 72, size=n), rng.integers(52, 70, size=n))
    birth_idx = conc_idx - age_at * 12 - rng.integers(0, 12, size=n)
    birth = pd.to_datetime([f"{i // 12}-{i % 12 + 1:02d}-{d:02d}" for i, d in zip(birth_idx, rng.integers(1, 28, size=n))])
    product = np.where(rng.random(n) < 0.6, "PA_LIFE", "PA_GUAR")
    guar = np.where(product == "PA_GUAR", rng.choice([60, 120], size=n), 0)
    premium = np.round(np.exp(rng.normal(np.log(15e6), 0.6, size=n)), -3)
    ind = rng.choice([0.0, 0.03, 0.05], p=[0.2, 0.3, 0.5], size=n)
    ac = rng.uniform(0.01, 0.03, size=n)
    load = rng.normal(0.06, 0.035, size=n)

    c = pd.DataFrame({
        "contract_id": [f"PA{100000 + i}" for i in range(n)], "product_code": product,
        "conclusion_date": conc, "payment_start_date": pay_start, "birth_date": birth, "gender": gender,
        "single_premium": premium, "indexation_rate": ind, "guarantee_months": guar,
        "acquisition_cost": np.round(premium * ac, 2), "_ac": ac, "_load": load,
    })
    c["initial_monthly_annuity"] = np.round(_price(c, cfg, curves), 2)

    # --- симуляция смертей и расторжений до конца периода данных ---
    a = cfg.assumptions[0]
    act_factor = np.where(gender == "M", 1.03, 1.00)
    death_idx = np.full(n, 10**7)
    cancel_idx = np.full(n, 10**7)
    reason = np.array([None] * n, dtype=object)
    for i in range(n):
        g = 1 if gender[i] == "F" else 0
        for m in range(conc_idx[i] + 1, b_idx + 1):
            age = (m - birth_idx[i]) // 12
            qx = a.mortality_table.qx[g, min(age, cfg.max_age)] * act_factor[i] * (1 - a.improvement) ** (m // 12 - a.base_year)
            if rng.random() < 1 - (1 - min(qx, 1)) ** (1 / 12):
                death_idx[i] = m
                break
    canc = rng.random(n) < 0.012
    for i in np.where(canc)[0]:
        lo = conc_idx[i] + 2
        if lo <= b_idx and death_idx[i] > lo:
            m = int(rng.integers(lo, min(b_idx, death_idx[i] - 1) + 1)) if min(b_idx, death_idx[i] - 1) >= lo else lo
            cancel_idx[i] = m
            reason[i] = rng.choice(["CANCELLATION", "TRANSFER"])
    g_end = s_idx + guar - 1

    death_date = [pd.Timestamp(f"{d // 12}-{d % 12 + 1:02d}-{int(x):02d}") if d < 10**7 else pd.NaT
                  for d, x in zip(death_idx, rng.integers(1, 28, size=n))]
    status, term_date = [], []
    for i in range(n):
        if cancel_idx[i] < 10**7:
            status.append("TERMINATED"); term_date.append(pd.Timestamp(month_end(cancel_idx[i])))
        elif death_idx[i] < 10**7:
            if g_end[i] > death_idx[i]:
                if g_end[i] <= b_idx:
                    status.append("TERMINATED"); term_date.append(pd.Timestamp(month_end(g_end[i]))); reason[i] = "GUARANTEE_END"
                else:
                    status.append("ACTIVE"); term_date.append(pd.NaT); reason[i] = None
            else:
                status.append("TERMINATED"); term_date.append(death_date[i]); reason[i] = "DEATH"
        else:
            status.append("ACTIVE"); term_date.append(pd.NaT)
    c["status"] = status
    c["death_date"] = death_date
    c["termination_date"] = term_date
    c["termination_reason"] = reason
    c["currency"] = "KZT"

    # --- фактические денежные потоки ---
    data_end = pd.Timestamp(month_end(b_idx))
    flows = []
    a0 = c["initial_monthly_annuity"].to_numpy()
    for i in range(n):
        cid = c.at[i, "contract_id"]
        flows.append((cid, "PREMIUM", conc[i], conc[i], premium[i]))
        flows.append((cid, "COMMISSION", conc[i], conc[i], c.at[i, "acquisition_cost"]))
        for m in range(s_idx[i], b_idx + 1):
            if m >= cancel_idx[i]:
                break
            alive_end = death_idx[i] > m
            in_g = m <= g_end[i]
            if not (alive_end or in_g):
                break
            amt = round(a0[i] * (1 + ind[i]) ** ((m - s_idx[i]) // 12), 2)
            due = pd.Timestamp(month_end(m))
            paid = due if rng.random() > 0.03 else due + pd.Timedelta(days=int(rng.integers(3, 10)))
            flows.append((cid, "ANNUITY", due, paid if paid <= data_end else pd.NaT, amt))
            e = a.per_policy_monthly * (1 + a.inflation) ** ((m - a.expense_base_idx) / 12) * rng.uniform(0.9, 1.1) + a.pct_of_payments * amt
            flows.append((cid, "EXPENSE", due, due, round(e, 2)))
        if cancel_idx[i] < 10**7:
            yrs = (cancel_idx[i] - conc_idx[i]) / 12
            sur = round(premium[i] * max(0.5, 0.95 - 0.04 * yrs), 2)
            d = pd.Timestamp(month_end(cancel_idx[i]))
            flows.append((cid, "SURRENDER", d, d, sur))
    f = pd.DataFrame(flows, columns=["contract_id", "cf_type", "due_date", "paid_date", "amount"])
    for col in ["due_date", "paid_date"]:
        f[col] = pd.to_datetime(f[col]).dt.strftime("%Y-%m-%d")

    c = c.drop(columns=["_ac", "_load"])
    for col in ["conclusion_date", "payment_start_date", "birth_date", "death_date", "termination_date"]:
        c[col] = pd.to_datetime(c[col]).dt.strftime("%Y-%m-%d")
    cols = ["contract_id", "product_code", "conclusion_date", "payment_start_date", "birth_date", "gender",
            "single_premium", "initial_monthly_annuity", "indexation_rate", "guarantee_months", "acquisition_cost",
            "status", "death_date", "termination_date", "termination_reason", "currency"]
    c[cols].to_csv(out_dir / "v_annuity_contracts.csv", index=False)
    f.to_csv(out_dir / "v_annuity_cashflows.csv", index=False)
    curve.to_csv(out_dir / "v_discount_curve.csv", index=False)
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="Генератор демо-данных")
    ap.add_argument("--contracts", type=int, default=2500)
    ap.add_argument("--from", dest="p_from", default="2024-01")
    ap.add_argument("--to", dest="p_to", default="2026-10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    mp = ROOT / "config" / "tables" / "mortality_demo.csv"
    if not mp.exists():
        write_mortality_table(mp)
    out = generate(args.out, args.contracts, args.p_from, args.p_to, args.seed)
    print(f"Демо-данные записаны в {out}")


if __name__ == "__main__":
    main()
