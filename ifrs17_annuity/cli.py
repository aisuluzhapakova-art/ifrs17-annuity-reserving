"""Командная строка.

    python -m ifrs17_annuity.cli run --period 2026-11
    python -m ifrs17_annuity.cli run-range --from 2024-01 --to 2026-10
    python -m ifrs17_annuity.cli report --period 2026-10
    python -m ifrs17_annuity.cli status

Коды возврата: 0 — успех, 1 — ошибка (данные, последовательность), 2 — не пройдены контроли.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import load_config
from .runner import DataQualityError, Runner, SequenceError


def _print_summary(summary) -> None:
    for r in summary.results:
        c = r.checks[r.checks["severity"] == "control"]
        print(f"  {r.period}: договоров {r.info['n_existing'] + r.info['n_new']:>6} (новых {r.info['n_new']:>4}), "
              f"групп {r.group_state['group_id'].nunique():>3}, контроли {int(c['passed'].sum())}/{len(c)}")
    warn = [i for i in summary.dq_issues if i["severity"] == "WARNING"]
    for i in warn:
        print(f"  ПРЕДУПРЕЖДЕНИЕ DQ: {i['check']}: {i['message']} ({i['count']})")
    if summary.excel_path:
        print(f"Excel: {summary.excel_path}")
    print(f"Время: {summary.seconds:.1f} с")
    if not summary.all_passed:
        print("НЕ ПРОЙДЕНЫ КОНТРОЛИ:")
        with pd.option_context("display.width", 200, "display.max_colwidth", 80):
            print(summary.failed_checks[["period", "group_id", "check", "diff"]].to_string(index=False))
    else:
        print("Все контроли пройдены.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ifrs17_annuity", description="МСФО 17: резервы по пенсионным аннуитетам")
    ap.add_argument("--config", default=None, help="путь к config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="закрытие месяца + Excel")
    p.add_argument("--period", required=True)
    p.add_argument("--no-excel", action="store_true")
    p = sub.add_parser("run-range", help="закрытие диапазона месяцев")
    p.add_argument("--from", dest="p_from", required=True)
    p.add_argument("--to", dest="p_to", required=True)
    p.add_argument("--no-excel", action="store_true")
    p = sub.add_parser("report", help="только Excel по рассчитанному месяцу")
    p.add_argument("--period", required=True)
    sub.add_parser("status", help="журнал расчетов")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    runner = Runner(cfg)
    try:
        if args.cmd == "run":
            s = runner.close_range(args.period, args.period, excel=not args.no_excel)
            _print_summary(s)
            return 0 if s.all_passed else 2
        if args.cmd == "run-range":
            s = runner.close_range(args.p_from, args.p_to, excel=not args.no_excel)
            _print_summary(s)
            return 0 if s.all_passed else 2
        if args.cmd == "report":
            print(f"Excel: {runner.write_excel(args.period)}")
            return 0
        if args.cmd == "status":
            log = runner.store.run_log()
            print(f"Хранилище: {cfg.results_url}")
            print(f"Закрытые месяцы: {', '.join(runner.store.closed_periods()) or 'нет'}")
            if len(log):
                with pd.option_context("display.width", 200, "display.max_colwidth", 60):
                    print(log.tail(30).to_string(index=False))
            return 0
    except (DataQualityError, SequenceError, ValueError, RuntimeError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
