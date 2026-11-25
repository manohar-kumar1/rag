create table sections (
  id bigint generated always as identity primary key,
  document_id bigint not null references documents (id) on delete cascade,
  section_index integer not null check (section_index >= 0),
  heading_path text not null default '',
  content text not null check (length(content) > 0),
  token_count integer not null check (token_count > 0),
  page_start integer check (page_start >= 1),
  page_end integer check (page_end >= page_start),
  unique (document_id, section_index)
);

alter table sections enable row level security;

alter table chunks
  add column section_id bigint references sections (id) on delete cascade,
  add column heading_path text not null default '',
  add column token_count integer check (token_count > 0),
  add column chunk_hash text;

create index idx_chunks_section on chunks (section_id);
create index idx_chunks_document_hash on chunks (document_id, chunk_hash);

alter table documents
  add column doc_type text,
  add column department text,
  add column product text,
  add column version text;

create index idx_documents_department on documents (department);
create index idx_documents_product on documents (product);
create index idx_documents_version on documents (version);
create index idx_documents_doc_type on documents (doc_type);

create table parse_cache (
  content_hash text not null,
  parser_key text not null,
  blocks jsonb not null,
  created_at timestamptz not null default now(),
  primary key (content_hash, parser_key)
);

alter table parse_cache enable row level security;
