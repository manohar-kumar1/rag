alter table documents
  add column processing_stage text
    constraint documents_processing_stage_check
    check (processing_stage in ('parsing', 'embedding'));

create table idempotency_keys (
  key text primary key check (length(key) between 1 and 255),
  request_hash text not null,
  state text not null check (state in ('in_progress', 'completed')),
  document_id bigint references documents (id) on delete cascade,
  created boolean,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null
);

create index idx_idempotency_keys_expires_at on idempotency_keys (expires_at);

alter table idempotency_keys enable row level security;
