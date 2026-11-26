-- Хранилище результатов в DWH (схема ifrs17_res).
-- Таблицы также создаются автоматически при первом запуске (SQLAlchemy metadata.create_all),
-- если у учетной записи есть права на DDL. Скрипт — для ручного развертывания администратором.

CREATE SCHEMA ifrs17_res;

CREATE TABLE ifrs17_res.contract_group (
    contract_id       VARCHAR(64) PRIMARY KEY,
    group_id          VARCHAR(128),
    portfolio         VARCHAR(64),
    cohort            INTEGER,
    profitability     VARCHAR(16),     -- ONEROUS / NO_SIG_RISK / OTHER
    recog_period      VARCHAR(7),      -- YYYY-MM
    premium           FLOAT,
    acquisition_cost  FLOAT,
    bel_init          FLOAT,
    ra_init           FLOAT,
    csm_init          FLOAT,           -- премия − IACF − BEL − RA (отрицательное = убыток)
    margin_pct        FLOAT
);

-- Состояние группы на конец месяца. Для перехода на МСФО 17 загрузить строки за месяц,
-- предшествующий первому расчетному (входящие CSM/LC/ПСД).
CREATE TABLE ifrs17_res.group_state (
    period         VARCHAR(7),
    group_id       VARCHAR(128),
    bel            FLOAT, bel_locked FLOAT, ra FLOAT, csm FLOAT, lc FLOAT, lic FLOAT,
    iacf_unamort   FLOAT, oci_accum FLOAT,
    cum_due_nonic  FLOAT, cum_due_ic FLOAT, cum_due_exp FLOAT, cum_paid FLOAT,
    PRIMARY KEY (period, group_id)
);

CREATE TABLE ifrs17_res.group_detail (
    period VARCHAR(7), contract_id VARCHAR(64), group_id VARCHAR(128),
    bel FLOAT, bel_locked FLOAT, ra FLOAT, status VARCHAR(24)
);

-- Строки отчетов в длинном формате: период × группа × отчет × строка × колонка.
CREATE TABLE ifrs17_res.report_lines (
    period VARCHAR(7), group_id VARCHAR(128), report VARCHAR(16), line VARCHAR(32), col VARCHAR(16), value FLOAT
);

CREATE TABLE ifrs17_res.checks (
    period VARCHAR(7), group_id VARCHAR(128), "check" VARCHAR(128), diff FLOAT, passed BOOLEAN,
    severity VARCHAR(16), info VARCHAR(256)
);

CREATE TABLE ifrs17_res.run_log (
    run_id INTEGER PRIMARY KEY, period VARCHAR(7), started_at TIMESTAMP, finished_at TIMESTAMP,
    status VARCHAR(16), n_contracts INTEGER, n_new INTEGER, checks_failed INTEGER,
    assumptions VARCHAR(64), config_hash VARCHAR(32), message VARCHAR(1000)
);
