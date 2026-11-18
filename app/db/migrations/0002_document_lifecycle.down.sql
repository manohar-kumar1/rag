drop index if exists idx_documents_processing;
drop index if exists idx_documents_pending;

alter table documents
  drop column if exists processed_at,
  drop column if exists processing_started_at,
  drop column if exists storage_key,
  drop column if exists size_bytes,
  drop column if exists content_type,
  drop column if exists original_filename,
  drop column if exists error,
  drop column if exists status;
