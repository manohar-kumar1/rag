from psycopg_pool import ConnectionPool

from app.core.readiness import Probe


def database_probe(pool: ConnectionPool, timeout_seconds: float) -> Probe:
    def probe() -> None:
        with pool.connection(timeout=timeout_seconds) as conn:
            conn.execute("select 1")

    return probe
