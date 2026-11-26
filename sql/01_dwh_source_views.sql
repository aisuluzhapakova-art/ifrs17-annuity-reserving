-- Шаблон витрин-источников в DWH для расчета МСФО 17 по пенсионным аннуитетам.
-- Синтаксис ANSI SQL; имена исходных таблиц (src_*) заменить на фактические.
-- Типы: даты — DATE, суммы — NUMERIC(18,2), ставки — NUMERIC(12,8).

-- 1. Договоры: один ряд на договор, последнее известное состояние.
CREATE OR REPLACE VIEW v_annuity_contracts AS
SELECT
    c.contract_no                         AS contract_id,
    c.product_code                        AS product_code,        -- PA_LIFE, PA_GUAR, ...
    c.conclusion_date                     AS conclusion_date,     -- дата заключения (признания)
    c.first_payment_date                  AS payment_start_date,
    p.birth_date                          AS birth_date,
    p.gender                              AS gender,              -- M / F
    c.single_premium                      AS single_premium,      -- сумма, переведенная из ЕНПФ
    c.initial_monthly_annuity             AS initial_monthly_annuity,
    c.indexation_rate                     AS indexation_rate,     -- доля, напр. 0.05
    COALESCE(c.guarantee_months, 0)       AS guarantee_months,
    COALESCE(c.commission_amount, 0)      AS acquisition_cost,
    c.status                              AS status,              -- ACTIVE / TERMINATED
    p.death_date                          AS death_date,
    c.termination_date                    AS termination_date,
    c.termination_reason                  AS termination_reason,  -- DEATH / GUARANTEE_END / CANCELLATION / TRANSFER
    c.currency                            AS currency             -- KZT
FROM src_annuity_contract c
JOIN src_person p ON p.person_id = c.annuitant_id
WHERE c.product_code IN ('PA_LIFE', 'PA_GUAR');

-- 2. Фактические денежные потоки (суммы > 0, направление задается типом).
CREATE OR REPLACE VIEW v_annuity_cashflows AS
SELECT
    t.contract_no                         AS contract_id,
    CASE t.operation_type
        WHEN 'PREM'  THEN 'PREMIUM'
        WHEN 'COMM'  THEN 'COMMISSION'
        WHEN 'ANN'   THEN 'ANNUITY'
        WHEN 'SURR'  THEN 'SURRENDER'      -- выкуп / перевод в другую СК
        WHEN 'EXP'   THEN 'EXPENSE'        -- атрибутируемые расходы, распределенные на договор
    END                                   AS cf_type,
    t.accrual_date                        AS due_date,   -- дата начисления
    t.payment_date                        AS paid_date,  -- дата оплаты (NULL, если не оплачено)
    ABS(t.amount)                         AS amount
FROM src_annuity_transaction t
WHERE t.operation_type IN ('PREM', 'COMM', 'ANN', 'SURR', 'EXP')
  AND t.is_reversed = 0;

-- 3. Бескупонная кривая ГЦБ KZT на конец каждого месяца (KASE / иной поставщик).
CREATE OR REPLACE VIEW v_discount_curve AS
SELECT
    k.curve_date                          AS curve_date,   -- последний день месяца
    k.tenor_years                         AS tenor_years,
    k.zero_rate                           AS zero_rate     -- годовая эффективная ставка, доля
FROM src_kzt_gov_zero_curve k
WHERE k.curve_date = LAST_DAY(k.curve_date);   -- в MS SQL: EOMONTH(k.curve_date)
