from collections.abc import Sequence

from app.answering.protocol import Turn
from app.db.source import ConnectionSource

RECENT_TURNS = (
    "select role, content from ("
    "select role, content, turn_index from conversation_turns where thread_id = %s "
    "order by turn_index desc limit %s) recent order by turn_index"
)
NEXT_INDEX = (
    "select coalesce(max(turn_index) + 1, 0) from conversation_turns where thread_id = %s"
)
LOCK_THREAD = "select pg_advisory_xact_lock(hashtextextended(%s, 0))"
INSERT_TURN = (
    "insert into conversation_turns (thread_id, turn_index, role, content) "
    "values (%s, %s, %s, %s)"
)
PRUNE = "delete from conversation_turns where created_at < now() - make_interval(secs => %s)"


class PostgresConversationStore:
    def __init__(self, source: ConnectionSource) -> None:
        self._source = source

    def recent(self, thread_id: str, limit: int) -> list[Turn]:
        with self._source.connection() as conn:
            rows = conn.execute(RECENT_TURNS, (thread_id, limit)).fetchall()
        return [Turn(role, content) for role, content in rows]

    def append(self, thread_id: str, turns: Sequence[Turn]) -> None:
        with self._source.connection() as conn, conn.transaction():
            conn.execute(LOCK_THREAD, (thread_id,))
            start = conn.execute(NEXT_INDEX, (thread_id,)).fetchone()[0]
            for offset, turn in enumerate(turns):
                conn.execute(INSERT_TURN, (thread_id, start + offset, turn.role, turn.content))

    def prune(self, older_than_seconds: float) -> int:
        with self._source.connection() as conn:
            return conn.execute(PRUNE, (older_than_seconds,)).rowcount
