alter table chunks add column context text;

drop index if exists idx_chunks_fts;
alter table chunks drop column fts;
alter table chunks add column fts tsvector generated always as (
  to_tsvector('{{fts_language}}', coalesce(context, '') || ' ' || content)
) stored;
create index idx_chunks_fts on chunks using gin (fts);
