drop table if exists idempotency_keys;

alter table documents drop column if exists processing_stage;
