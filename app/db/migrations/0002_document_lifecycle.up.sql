alter table documents
  add column status text not null default 'pending'
    constraint documents_status_check
    check (status in ('pending', 'processing', 'ready', 'failed')),
  add column error text,
  add column original_filename text,
  add column content_type text,
  add column size_bytes bigint check (size_bytes >= 0),
  add column storage_key text unique,
  add column processing_started_at timestamptz,
  add column processed_at timestamptz;

update documents d
  set status = 'ready', processed_at = d.created_at
  where exists (select 1 from chunks c where c.document_id = d.id);

create index idx_documents_pending
  on documents (created_at) where status = 'pending';

create index idx_documents_processing
  on documents (processing_started_at) where status = 'processing';
