# Migration to dbt — step-by-step guide

This document describes a practical path to **adopt dbt** when you already have **Snowflake** (and optionally **ELT loaders**, **Salesforce**, or **integration stored procedures**). It is not a single “big bang” cutover; it is usually **phased** so analytics and quality improve before you touch risky operational pipelines.

**Related docs:** [Salesforce + Snowflake + dbt](README_Salesforce_Snowflake.md), example project [`my_dbt/README.md`](my_dbt/README.md).

---

## Where the migration happens (two directories)

A migration has a **source** (existing Snowflake integration scripts) and a **target** (the dbt project where new models live). They are **not** the same folder.

### Source — existing integration SQL (U8)

This is the directory that holds **today’s** Snowflake stored procedures and related scripts you are assessing or gradually replacing with dbt-friendly patterns:

| Role | Path |
|------|------|
| **Integration / procedures (read here)** | `/Users/thongbui/open_issue/U8_WHSE/P1062_U8_SN0_to_UpdateGitSnowflake/STAGING/U8_INTEGRATIONS` |

Typical layout: `procedures/*.sql` — `MERGE`/`UPDATE` flows, snapshot `CLONE` jobs, ownership helpers, ops logging (`INT_OPS`, `MSG_OPS`). **You do not run `dbt` from this directory.** Use it for **Phase 0** inventory, naming conventions (`DS*_II`, `*_MI`), and deciding what stays as procedures vs what becomes dbt models.

### Target — dbt project (run CLI here)

In **this git repository** (`python_project`), the dbt project you migrate **into** lives here:

| Role | Path | Notes |
|------|------|--------|
| **dbt project root** | `dbt/my_dbt/` (from repo root) | `dbt_project.yml`, `models/`, `tests/`, `seeds/`. **Run every `dbt` command from this directory.** |
| **Guides** | `dbt/` | `README_Migration_to_dbt.md`, `README_Salesforce_Snowflake.md` — documentation only. |
| **Python env (optional)** | `dbt/venv312/` | Activate before `dbt` if you use this venv (`my_dbt/README.md`). |

Example — **from `python_project` repo root**:

```bash
cd dbt/my_dbt
source ../venv312/bin/activate   # optional
dbt debug
```

If you scaffold another project with `dbt init`, keep one documented **target** path for the team (for example still under `dbt/<name>/`).

### Flow

1. **Edit / review** legacy logic under `.../U8_INTEGRATIONS` (or deploy procedures from that tree to Snowflake).
2. **Author and run** dbt models under `python_project/dbt/my_dbt/` (or your chosen dbt root).
3. Point dbt **`sources.yml`** at the Snowflake tables **produced by** those procedures, not at the `.sql` files on disk.

---

## What “convert to dbt” usually means

dbt is best at **versioned transformations** in the warehouse: **SQL models**, **tests**, **documentation**, and **lineage**. It does **not** replace every Snowflake **stored procedure** (cursors, multi-step transactions, dynamic `CLONE`, or email queues). A typical migration **moves read-only analytics and reporting SQL into dbt**, and **keeps** procedural integration jobs in Snowflake **until** you deliberately refactor them.

---

## Migration phases at a glance

```mermaid
flowchart TB
  P0[Assess]
  P1[Scaffold dbt project]
  P2[profiles and targets]
  P3[Define sources]
  P4[Staging models]
  P5[Intermediate and marts]
  P6[Tests and docs]
  P7[CI and environments]
  P8[Orchestrate and cutover]

  P0 --> P1
  P1 --> P2
  P2 --> P3
  P3 --> P4
  P4 --> P5
  P5 --> P6
  P6 --> P7
  P7 --> P8
```

---

## Phase 0 — Assess and scope

1. **Inventory existing SQL** — List views, ad-hoc reports, BI extracts, and scheduled SQL that **only read** warehouse tables and could become **dbt models**.
2. **Inventory what must stay** — Stored procedures that **loop**, **call other procedures**, manage **transactions** and **rollback**, or write **ops/notification** tables (`MSG_OPS`, `INT_OPS`) often stay **outside** dbt until you have a strong reason to change them.
3. **Pick boundaries** — Decide: “dbt owns everything **after** schema `X`” or “dbt owns **marts** only; raw and integration stay as-is.”
4. **Naming targets** — Align **dev** and **prod** databases/schemas (for example `_DEV` suffix vs production) with dbt **targets** in `profiles.yml`.
5. **Stakeholders** — Agree who approves model changes (data team) vs who runs integration jobs (ops).

---

## Phase 1 — Scaffold the dbt project

1. **Install** the adapter for your warehouse, for example:

   ```bash
   pip install "dbt-snowflake>=1.7,<2"
   ```

2. **Create a project** (or copy a starter template). Run `dbt init` from the **parent** of where you want the project folder, then `cd` into the new project directory (or use the existing **`dbt/my_dbt/`** directory in this repo):

   ```bash
   cd dbt
   dbt init your_project_name
   cd your_project_name
   ```

3. **Set** `project_name`, `profile`, and paths in **`dbt_project.yml`** inside that project directory (`model-paths`, `test-paths`, `seed-paths`, …).
4. **Folder convention** (common): `models/staging/`, `models/intermediate/`, `models/marts/`; keep **one concern per layer** so newcomers navigate easily.

---

## Phase 2 — Connect Snowflake (`profiles.yml`)

1. Create a **profile** per project (often `~/.dbt/profiles.yml` or environment variables in CI).
2. Define **outputs** for at least **`dev`** and **`prod`** (or `prod` and `ci`): different **database**, **schema**, **warehouse**, **role**, or **credentials**.
3. Prefer **key-pair auth** for service users; avoid committing secrets to git.
4. From your **dbt project directory** (e.g. `dbt/my_dbt`), run:

   ```bash
   dbt debug
   ```

   and fix connection issues before modeling.

---

## Phase 3 — Declare sources (`sources.yml`)

1. **List** tables or views that **dbt reads but does not own** (raw loads, integration tables, Salesforce mirrors).
2. For each **source**, set `database`, `schema`, and `tables` in `models/.../sources.yml`.
3. Use **`source('name','table')`** in staging models instead of hard-coded `database.schema.table` (easier environment moves and renames).
4. Optionally add **freshness** and **descriptions** for critical tables.

---

## Phase 4 — Build staging models

1. **One staging model per source entity** (or per logical group) — thin layer: **rename**, **cast**, **dedupe keys**, **standardize** column names.
2. **Materialization** — Often **`view`** for staging to reduce storage churn; use **`table`** if performance requires it.
3. **Do not** duplicate heavy integration logic here if it already lives in procedures; **stage what the integration layer already produced**.

---

## Phase 5 — Intermediate and marts

1. **Intermediate** — Joins and business rules that are **reused** by multiple marts.
2. **Marts** — Fact/dimension or wide tables for **BI** and **metrics**; stable contracts for downstream tools.
3. **Use `ref()`** only between **dbt models**; use **`source()`** for raw/integration inputs.
4. **Materialization** — Often **`table`** or **`incremental`** for large marts (choose strategy with your warehouse’s strengths).

---

## Phase 6 — Tests and documentation

1. **Generic tests** in YAML — `not_null`, `unique`, `relationships`, `accepted_values` on keys and critical fields.
2. **Singular tests** — Custom SQL in `tests/` for **business rules** that cannot be expressed as generic tests.
3. **Unit tests** (optional) — For logic-heavy models with **mocked** inputs, per dbt docs.
4. **`dbt docs generate`** — Add **descriptions** to models and columns so lineage and docs stay useful.

---

## Phase 7 — CI/CD and environments

1. **Git** — Store the dbt project in a repository; **main** or **master** reflects production-approved code.
2. **CI** — On pull requests, run **`dbt parse`**, **`dbt compile`**, and optionally **`dbt build --select state:modified+`** when artifacts are available.
3. **Secrets** — Inject **profiles** via CI secrets or env vars; never commit passwords or keys.
4. **dbt Cloud** (optional) — Hosted runs, scheduling, and alerts if you do not want to run **Airflow** or **Kubernetes** jobs yourself.

---

## Phase 8 — Orchestration and cutover

1. **Order of execution** — Ensure **ingestion** and **integration procedures** finish **before** `dbt run` (or `dbt build`) so sources are fresh. Use **Airflow**, **Snowflake Tasks**, **dbt Cloud jobs**, or similar.
2. **Cutover BI** — Point dashboards from **legacy views** to **dbt-built** tables/views (or swap views in a thin bridge layer).
3. **Monitor** — Watch **dbt run** failures, **test** failures, and warehouse **cost** after cutover.

---

## Snowflake stored procedures — what to convert vs keep

| Pattern | Typical approach |
|--------|-------------------|
| **Procedural merges** (`UPDATE`/`MERGE` with loops, `CALL` chains, `INT_OPS`) | **Keep** in procedures or Tasks initially; **expose** outputs to dbt via **sources** or **views**. |
| **Dynamic CLONE snapshots** across many tables | **Keep** outside dbt; dbt **snapshots** are not a drop-in replacement for bulk **CLONE**. |
| **JavaScript** procedures (grants, ownership) | **Governance**; keep outside dbt. |
| **Reporting SQL** (aggregations, joins for BI) | **Move** into dbt **staging/marts**. |
| **Ad-hoc SQL** in spreadsheets or notebooks | **Replace** with dbt models over time. |

---

## Validation checklist before calling the migration “done”

- [ ] **Source scripts** are tracked (for example under `.../STAGING/U8_INTEGRATIONS`) and **dbt** is run only from the **dbt project root** (here: `dbt/my_dbt/`).
- [ ] `dbt debug` succeeds for **dev** and **prod** targets.
- [ ] All critical **sources** documented; **staging** models cover agreed entities.
- [ ] **Marts** match or improve on legacy **row counts** / **spot checks** for key metrics.
- [ ] **Tests** run in CI and on schedule; failures are **actionable**.
- [ ] **Orchestration** order is **ingest → integration → dbt** (or documented exception).
- [ ] **Owners** and **on-call** for dbt failures are defined.

---

## Quick reference commands

Run from the **dbt project root** (in this repo: `dbt/my_dbt`). Legacy procedures live under `/Users/thongbui/open_issue/U8_WHSE/P1062_U8_SN0_to_UpdateGitSnowflake/STAGING/U8_INTEGRATIONS` — not here.

```bash
cd dbt/my_dbt
dbt debug
dbt run --select staging
dbt test
dbt build
dbt docs generate && dbt docs serve
```

---

## Further reading

- [dbt best practices](https://docs.getdbt.com/best-practices)
- [dbt + Snowflake](https://docs.getdbt.com/docs/core/connect-data-platform/snowflake-setup)
- [Salesforce + Snowflake + dbt overview](README_Salesforce_Snowflake.md)
