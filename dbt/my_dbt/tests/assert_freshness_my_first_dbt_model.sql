-- Singular test: fails if the latest ingested_at is older than 48 hours.
-- Run after dbt run so ingested_at reflects the last build.
select 1 as stale
from (
    select max(ingested_at) as max_ingested
    from {{ ref('my_first_dbt_model') }}
)
where max_ingested < timestamp_sub(current_timestamp(), interval 48 hour)
