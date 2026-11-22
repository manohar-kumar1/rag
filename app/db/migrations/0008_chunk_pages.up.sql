alter table chunks
  add column page_start integer check (page_start >= 1),
  add column page_end integer check (page_end >= page_start);
