create table answer_cache (
  id bigint generated always as identity primary key,
  scope text not null,
  question_key text not null,
  corpus_version text not null,
  question text not null,
  embedding extensions.vector({{embedding_dimension}}),
  answer jsonb not null,
  hits integer not null default 0,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null,
  unique (scope, question_key, corpus_version)
);

create index idx_answer_cache_expires_at on answer_cache (expires_at);

alter table answer_cache enable row level security;
