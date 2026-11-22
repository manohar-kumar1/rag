alter table chunks
  drop column if exists page_end,
  drop column if exists page_start;
