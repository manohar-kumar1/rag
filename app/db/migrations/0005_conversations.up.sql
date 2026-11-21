create table conversation_turns (
  thread_id uuid not null,
  turn_index integer not null check (turn_index >= 0),
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  created_at timestamptz not null default now(),
  primary key (thread_id, turn_index)
);

create index idx_conversation_turns_created_at on conversation_turns (created_at);

alter table conversation_turns enable row level security;
