create function set_updated_at() returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

create table documents (
  id bigint generated always as identity primary key,
  source text not null check (length(source) > 0),
  title text,
  content_hash text not null unique,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_documents_updated_at
  before update on documents
  for each row execute function set_updated_at();

create table chunks (
  id bigint generated always as identity primary key,
  document_id bigint not null references documents (id) on delete cascade,
  chunk_index integer not null check (chunk_index >= 0),
  content text not null check (length(content) > 0),
  embedding extensions.vector({{embedding_dimension}}) not null,
  fts tsvector generated always as (to_tsvector('{{fts_language}}', content)) stored,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index idx_chunks_fts on chunks using gin (fts);

create index idx_chunks_embedding_hnsw
  on chunks using hnsw (embedding extensions.vector_cosine_ops);

alter table documents enable row level security;
alter table chunks enable row level security;
