import logging
from collections.abc import Callable, Mapping

from app.exceptions.exception import NotReady

logger = logging.getLogger(__name__)

Probe = Callable[[], None]


class ReadinessChecker:
    def __init__(self, probes: Mapping[str, Probe]) -> None:
        self._probes = dict(probes)

    def check(self) -> dict[str, str]:
        failed = []
        for name, probe in self._probes.items():
            try:
                probe()
            except Exception:
                logger.exception("readiness probe %s failed", name)
                failed.append(name)
        if failed:
            raise NotReady(failed)
        return {name: "ok" for name in self._probes}
