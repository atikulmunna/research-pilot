"""Background execution of projects for the API and dashboard."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Callable, Dict, Tuple

from .config import Settings
from .orchestrator import Orchestrator
from .state.store import ProjectStore

OrchestratorFactory = Callable[[ProjectStore, Event], Orchestrator]


class ProjectRunner:
    def __init__(self, settings: Settings, factory: OrchestratorFactory | None = None, max_workers: int = 2):
        self.settings = settings
        self.factory = factory or (lambda store, cancel: Orchestrator(settings, store, cancel_event=cancel))
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._active: Dict[str, Tuple[Future, Event]] = {}
        self._lock = Lock()

    def start(self, project_id: str, max_steps: int | None = None) -> bool:
        with self._lock:
            current = self._active.get(project_id)
            if current and not current[0].done():
                return False
            cancel = Event()
            store = ProjectStore.open(self.settings.workspace_dir, project_id)
            future = self._pool.submit(lambda: self.factory(store, cancel).run(max_steps))
            self._active[project_id] = (future, cancel)
            return True

    def cancel(self, project_id: str) -> bool:
        with self._lock:
            current = self._active.get(project_id)
            if not current or current[0].done():
                return False
            current[1].set()
            return True

    def is_running(self, project_id: str) -> bool:
        with self._lock:
            current = self._active.get(project_id)
            return bool(current and not current[0].done())

    def shutdown(self) -> None:
        with self._lock:
            for _, cancel in self._active.values():
                cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
