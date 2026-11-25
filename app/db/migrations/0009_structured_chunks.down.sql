drop table if exists parse_cache;

drop index if exists idx_documents_doc_type;
drop index if exists idx_documents_version;
drop index if exists idx_documents_product;
drop index if exists idx_documents_department;

alter table documents
  drop column if exists version,
  drop column if exists product,
  drop column if exists department,
  drop column if exists doc_type;

drop index if exists idx_chunks_document_hash;
drop index if exists idx_chunks_section;

alter table chunks
  drop column if exists chunk_hash,
  drop column if exists token_count,
  drop column if exists heading_path,
  drop column if exists section_id;

drop table if exists sections;
