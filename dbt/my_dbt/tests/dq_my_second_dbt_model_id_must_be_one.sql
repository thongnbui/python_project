-- Singular test: fails if any row has id other than 1 (business rule for this model).
select *
from {{ ref('my_second_dbt_model') }}
where id != 1
