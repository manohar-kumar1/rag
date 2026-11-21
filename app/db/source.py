from contextlib import AbstractContextManager
from typing import Protocol

import psycopg


class ConnectionSource(Protocol):
    def connection(self) -> AbstractContextManager[psycopg.Connection]: ...
