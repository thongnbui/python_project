# my_dbt

This directory is a **dbt Core project** named `my_dbt`. dbt turns SQL files in `models/` into tables or views in your warehouse and lets you document and test them. This repo ships with the usual starter **example** models so you can run `dbt run` and `dbt test` end-to-end once your profile is configured.

## How it fits together

1. **`dbt_project.yml`** defines the project name (`my_dbt`), which **profile** to use (`profile: my_dbt`), and where dbt looks for models, seeds, tests, macros, and snapshots.
2. **Models** live under `models/`. Each `.sql` file is one model. dbt compiles Jinja (e.g. `{{ config(...) }}`, `{{ ref('...') }}`) and runs the resulting SQL against your warehouse.
3. **`models/example/schema.yml`** attaches descriptions and **generic tests** (e.g. `unique`, `not_null`) to those models.

Warehouse credentials are **not** in this repo. They live in your dbt profile (typically `~/.dbt/profiles.yml` under the `my_dbt` profile). For BigQuery setup notes, see the parent doc at `../README_BigQuery.md`.

## Layout

| Path | Role |
|------|------|
| `dbt_project.yml` | Project name, profile, paths, default model config |
| `models/example/*.sql` | Example models (see below) |
| `models/example/schema.yml` | Model/column docs and data tests |
| `models/example/sources.yml` | [Sources](https://docs.getdbt.com/docs/build/sources): external tables + **freshness** |
| `seeds/raw_example_events.csv` | Demo seed for `source()` / `dbt source freshness` |
| `analyses/` | Ad-hoc analysis SQL (compiled only; not run as pipeline models) |
| `tests/` | Custom (singular) tests and related SQL |
| `seeds/` | CSV files loaded with `dbt seed` |
| `macros/` | Reusable Jinja (`{% macro %}`) |
| `snapshots/` | [Snapshot](https://docs.getdbt.com/docs/build/snapshots) definitions (SCD-style tables) |
| `target/` | Compiled SQL and artifacts (generated; safe to delete / `dbt clean`) |
| `dbt_packages/` | Installed packages (from `dbt deps`; cleaned with `dbt clean`) |

Paths in the middle rows come from `dbt_project.yml` (`model-paths`, `analysis-paths`, etc.). This project uses **`models/`**, **`tests/`**, **`seeds/`**, and **`models/example/sources.yml`**; other dirs are optional placeholders.

## How `models/example/` is used

- **Discovery** — dbt walks everything under `model-paths` (here `models/`). Each **`*.sql`** file is a **model**: a node in the DAG that `dbt run` / `dbt build` can materialize. The **`example/`** segment is only a subdirectory for organization; it could be renamed (e.g. `staging/`, `marts/`) and you would update `dbt_project.yml` if you still want folder-specific config under `models: my_dbt: …`.

- **Config mapping** — In `dbt_project.yml`, `models: my_dbt: example:` applies settings to **all models whose path is under** `models/example/`. Here that sets `+materialized: view` by default. A model file can override that with `{{ config(...) }}` at the top (as `my_first_dbt_model` does for `table`).

- **YAML next to models** — Files like **`schema.yml`** (any name `*.yml` / `*.yaml` under `models/`) are **not** executed as SQL. They register **metadata**: model descriptions, column descriptions, and **tests** (`unique`, `not_null`, relationships, etc.). dbt associates `schema.yml` entries with models by **`name:`** matching the model name.

- **Build order** — Models under `example/` use **`ref()`** like any other folder; dbt only cares about the dependency graph, not the folder name.

## Other project folders (analyses, tests, seeds, macros, snapshots)

These directories are declared in `dbt_project.yml` so dbt knows where to look. They behave differently from `models/`:

| Folder | Purpose | Typical command |
|--------|---------|-----------------|
| **`analyses/`** | Exploratory or one-off SQL that uses `ref()` / macros. Compiled with **`dbt compile`** (output under `target/`) but **not** scheduled as pipeline models by `dbt run`. | `dbt compile` |
| **`tests/`** | **Singular** tests: custom `.sql` files that return failing rows. (Generic tests like `unique` usually live in `schema.yml` next to models.) | `dbt test` |
| **`seeds/`** | Small reference **CSV** files checked into git; **`dbt seed`** loads them into the warehouse as tables you can **`ref()`** from models. | `dbt seed` |
| **`macros/`** | **Jinja macros** shared across models, tests, and analyses. | (used at compile time) |
| **`snapshots/`** | **`snapshot`** blocks for slowly changing dimensions / change data capture. | `dbt snapshot` |

Optional folders stay empty until you add files; this project also uses **`seeds/`** and **`tests/`** as described above.

## Configuration

- **Global model defaults** for this project are under `models: my_dbt: ...` in `dbt_project.yml`.
- Everything under `models/example/` is configured as **`view`** by default (`+materialized: view`).
- **Per-model overrides** use `{{ config(...) }}` at the top of a model file. In this project, `my_first_dbt_model` sets `materialized='table'`, so that model builds as a **table** even though the folder default is views.

## Example models

### `my_first_dbt_model`

- Builds a small inline dataset (including a null `id` filtered out with `where id is not null`).
- Adds **`ingested_at`** with **`current_timestamp()`** for recency checks.
- Materializes as a **table** via `{{ config(materialized='table') }}`.

### `my_second_dbt_model`

- Selects **`id`** from the first model using **`{{ ref('my_first_dbt_model') }}`** and filters to `id = 1`.
- dbt enforces **build order**: the first model runs before the second.

### Tests in `schema.yml`

Generic **data tests** on `id` include `not_null`, `unique`, `accepted_values`, and (on the second model) **`relationships`** to the first model. See **`tests/`** for singular SQL tests.

### Freshness

Two patterns are configured:

1. **Source freshness** ([docs](https://docs.getdbt.com/docs/deploy/source-freshness)) — In **`models/example/sources.yml`**, the source **`example.raw_example_events`** points at the seed table **`raw_example_events`** (same target database/schema). **`config.loaded_at_field`** is **`loaded_at`**; **`config.freshness`** sets **`warn_after`** / **`error_after`** vs `max(loaded_at)` at run time. Load the seed first, then run:

   ```bash
   dbt seed
   dbt source freshness
   ```

   **`warn_after`** uses a short window so stale demo data often **warns**; **`error_after`** is set very wide so the sample CSV does not fail CI as it ages—**tighten both in production** and keep **`loaded_at`** realistic (or refresh the seed).

2. **Model recency (singular test)** — **`tests/assert_freshness_my_first_dbt_model.sql`** fails if **`max(ingested_at)`** on **`my_first_dbt_model`** is older than **48 hours** (run after **`dbt run`** so the table exists). Adjust the interval in that file to match your SLA.

### Unit tests

[Unit tests](https://docs.getdbt.com/docs/build/unit-tests) mock upstream **`ref()`** inputs and assert the model’s output without relying on production data. This project defines them in **`models/example/schema.yml`** under the top-level **`unit_tests:`** key (required in current dbt versions; nesting under `models:` is deprecated).

They target **`my_second_dbt_model`**, which depends on **`ref('my_first_dbt_model')`**: one case keeps only `id = 1`, another expects **no rows** when upstream has no `id = 1`.

Run only unit tests:

```bash
dbt test --select resource_type:unit_test
```

`my_first_dbt_model` has no `ref()` inputs to mock, so it is covered by **data tests** and inline SQL rather than unit tests here.

## Python environment (dbt CLI)

dbt for this repo is meant to run from the **Python 3.12 virtualenv** under the parent **`dbt/`** directory, not the project-root `venv312`:

| Location | Path |
|----------|------|
| Virtualenv | `dbt/venv312` |

Activate it **before** `dbt` commands (from `dbt/my_dbt`):

```bash
source ../venv312/bin/activate
```

## Common commands

Run from this directory (`dbt/my_dbt`), with the venv above activated:

```bash
dbt deps          # if you add packages later
dbt parse         # validate project compiles
dbt compile       # render Jinja to SQL; write under target/ (no warehouse execution)
dbt seed          # load CSV seeds (needed for source freshness demo)
dbt source freshness   # check max(loaded_at) vs warn_after / error_after on sources
dbt run           # build models
dbt test          # data tests + unit tests (see unit test selection below)
dbt test --select resource_type:unit_test   # unit tests only
dbt build         # run + test in dependency order
dbt docs generate && dbt docs serve   # optional: lineage and docs site
```

Ensure the **`my_dbt`** profile in `~/.dbt/profiles.yml` points at your warehouse (project, dataset, credentials, etc.).

## Compile, select, and target

Many dbt commands accept **selection** (`-s` / `--select`) and **target** (`-t` / `--target`). Example:

```bash
dbt compile -s table_b -t prod
```

- **`dbt compile`** — Expands Jinja (`{{ ref() }}`, `{{ config() }}`, macros, etc.) and writes compiled SQL under `target/`. It does **not** connect to the warehouse to create or replace relations; use it to inspect SQL or catch errors before `dbt run` / `dbt build`.

- **`-s table_b`** (`--select`) — Limits which nodes are included. `table_b` is a [selector](https://docs.getdbt.com/reference/node-selection/syntax): usually a model name or path-style selector. If nothing matches, dbt reports a selection error.

- **`-t prod`** (`--target`) — Chooses the named **output** under your profile (e.g. in `~/.dbt/profiles.yml` for `profile: my_dbt`). Typical setups define `dev` and `prod` with different projects, datasets, schemas, or credentials. Omitting `-t` uses the profile’s default `target`.

**Contrast:** `dbt run -s table_b -t prod` uses the same selection and target but **executes** the SQL and materializes objects in the warehouse for `prod`.

## Summary

- **Project layout** — `dbt_project.yml` drives the project; models live under `models/example/` (`.sql` + `schema.yml`); `models/example/sources.yml` defines the `raw_example_events` source and its **freshness** (backed by the seed).
- **Tests** — `tests/` includes singular SQL checks, including **48-hour** recency on `ingested_at` for `my_first_dbt_model`.
- **Setup** — Activate `dbt/venv312` and configure the `my_dbt` profile in `~/.dbt/profiles.yml`.
- **Commands** — Use `dbt seed`, `dbt source freshness`, `dbt build`, or `dbt test` as needed.
