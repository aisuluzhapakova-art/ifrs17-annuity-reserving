"""Регрессионные тесты: тождества и сценарии (раздел 12 PRD)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.generate_demo_data import generate, write_mortality_table  # noqa: E402
from ifrs17_annuity.config import load_config  # noqa: E402
from ifrs17_annuity.runner import Runner  # noqa: E402
from ifrs17_annuity.store import ResultStore  # noqa: E402

N_CONTRACTS = 400
P_FROM, P_TO = "2024-01", "2026-02"


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    mp = ROOT / "config" / "tables" / "mortality_demo.csv"
    if not mp.exists():
        write_mortality_table(mp)
    d = tmp_path_factory.mktemp("demo")
    cfg = load_config(ROOT / "config" / "config.yaml")
    generate(d, n_contracts=N_CONTRACTS, p_from=P_FROM, p_to=P_TO, seed=7, cfg=cfg)
    return d


def make_runner(demo_dir, tmp_path, **policy) -> Runner:
    cfg = load_config(ROOT / "config" / "config.yaml")
    cfg.raw["source"] = {**cfg.raw["source"], "type": "csv", "csv_dir": str(demo_dir)}
    cfg.raw["output_dir"] = str(tmp_path)
    if policy:
        cfg = cfg.with_overrides(**policy)
    store = ResultStore("sqlite:///" + (tmp_path / "results.db").as_posix())
    return Runner(cfg, store=store)


def assert_all_passed(summary):
    failed = summary.failed_checks
    assert summary.all_passed, f"Не пройдены контроли:\n{failed.to_string()}"


@pytest.mark.parametrize("ifie", ["OCI", "PL"])
def test_full_run_all_checks_pass(demo_dir, tmp_path, ifie):
    r = make_runner(demo_dir, tmp_path, ifie_disaggregation=ifie)
    s = r.close_range(P_FROM, P_TO, excel=(ifie == "OCI"))
    assert len(s.results) == 26
    assert_all_passed(s)
    gs = r.store.group_state(P_TO)
    assert (gs["csm"] >= -0.5).all()
    assert len(gs) >= 6
    if ifie == "PL":
        assert (gs["oci_accum"].abs() < 1e-6).all()
    else:
        assert s.excel_path and Path(s.excel_path).exists()


def test_assumption_change_adjusts_csm(demo_dir, tmp_path):
    r = make_runner(demo_dir, tmp_path)
    r.close_range("2024-01", "2025-12", excel=False)
    lines = r.store.report_lines("2025-11", "2025-12")
    aoc = lines[(lines["report"] == "AOC") & (lines["line"] == "S4")]
    nov = aoc[(aoc["period"] == "2025-11") & (aoc["col"] == "BEL_LOCKED")]["value"].abs().sum()
    dec = aoc[(aoc["period"] == "2025-12") & (aoc["col"] == "BEL_LOCKED")]["value"].sum()
    assert nov < 1e-6, "В ноябре допущения не менялись"
    assert dec > 0, "Снижение смертности должно увеличить BEL"
    fs = lines[(lines["report"] == "AOC_CSM") & (lines["line"] == "FS") & (lines["period"] == "2025-12")]["value"].sum()
    lc = lines[(lines["report"] == "AOC_LC") & (lines["line"] == "FS") & (lines["period"] == "2025-12")]["value"].sum()
    assert fs < 0 or lc > 0, "Неблагоприятное изменение уменьшает CSM или создает LC"


def test_lrc_at_recognition(demo_dir, tmp_path):
    """В первом месяце выплат еще нет: ООП = премия − IACF + убыток по обременительным договорам."""
    r = make_runner(demo_dir, tmp_path)
    s = r.close_range("2024-01", "2024-01", excel=False)
    assert_all_passed(s)
    gs = r.store.group_state("2024-01")
    cg = r.store.contract_groups()
    lrc = (gs["bel"] + gs["ra"] + gs["csm"]).sum()
    loss = gs["lc"].sum()
    expected = cg["premium"].sum() - cg["acquisition_cost"].sum() + loss
    assert abs(lrc - expected) < 1.0
    assert abs(loss + cg.loc[cg["csm_init"] < 0, "csm_init"].sum()) < 1.0
    assert (gs["lic"].abs() < 1e-6).all()


def test_late_reported_death(demo_dir, tmp_path):
    """Смерть, отраженная в DWH с опозданием, снимает обязательство через шаг 0a без нарушения контролей."""
    r = make_runner(demo_dir, tmp_path)
    r.close_range("2024-01", "2025-06", excel=False)
    det = r.store.group_detail("2025-06")
    cg = r.store.contract_groups()
    c = r.source().contracts
    cand = c[c["death_date"].isna() & c["termination_date"].isna() & (c["guarantee_months"] == 0)
             & c["contract_id"].isin(det.loc[det["status"] == "ALIVE", "contract_id"])]
    cid = cand["contract_id"].iloc[0]
    bel_before = float(det.loc[det["contract_id"] == cid, "bel"].iloc[0])
    assert bel_before > 0

    # исправление данных: смерть в мае 2025, отражена в DWH только сейчас
    sd = r.source()
    sd.contracts.loc[sd.contracts["contract_id"] == cid, ["death_date", "status", "termination_date",
                                                          "termination_reason"]] = \
        [pd.Timestamp("2025-05-15"), "TERMINATED", pd.Timestamp("2025-05-15"), "DEATH"]
    r._prepared = None
    s = r.close_range("2025-07", "2025-07", excel=False)
    assert_all_passed(s)
    grp = cg.loc[cg["contract_id"] == cid, "group_id"].iloc[0]
    chk = r.store.checks("2025-07")
    s0a = chk[(chk["group_id"] == grp) & chk["check"].str.startswith("6.")]["diff"].iloc[0]
    assert s0a < -0.5 * bel_before
    det7 = r.store.group_detail("2025-07")
    assert abs(float(det7.loc[det7["contract_id"] == cid, "bel"].iloc[0])) < 1e-6


def test_rerun_and_sequence(demo_dir, tmp_path):
    from ifrs17_annuity.runner import SequenceError

    r = make_runner(demo_dir, tmp_path)
    r.close_range("2024-01", "2024-04", excel=False)
    with pytest.raises(SequenceError):
        r.close_month("2024-06")
    before = r.store.group_state("2024-04")["csm"].sum()
    r.close_month("2024-03")              # перезапуск удаляет март и апрель
    assert r.store.last_closed_period() == "2024-03"
    r.close_month("2024-04")
    after = r.store.group_state("2024-04")["csm"].sum()
    assert abs(before - after) < 1e-6


def test_ra_disaggregation_policy(demo_dir, tmp_path):
    r = make_runner(demo_dir, tmp_path, ra_disaggregation=True)
    s = r.close_range("2024-01", "2024-12", excel=False)
    assert_all_passed(s)
