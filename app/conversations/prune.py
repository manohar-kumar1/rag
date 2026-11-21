from app.conversations.store import PostgresConversationStore
from app.core.config import get_settings
from app.db.connection import create_pool


def main() -> int:
    settings = get_settings()
    pool = create_pool(settings)
    pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
    try:
        removed = PostgresConversationStore(pool).prune(settings.conversation_retention_seconds)
    finally:
        pool.close()
    print(f"removed {removed} conversation turns older than the retention period")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
