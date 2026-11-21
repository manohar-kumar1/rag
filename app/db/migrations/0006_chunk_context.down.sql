drop index if exists idx_chunks_fts;
alter table chunks drop column fts;
alter table chunks add column fts tsvector generated always as (
  to_tsvector('{{fts_language}}', content)
) stored;
create index idx_chunks_fts on chunks using gin (fts);

alter table chunks drop column if exists context;
