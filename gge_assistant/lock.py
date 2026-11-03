from __future__ import annotations

import os
from pathlib import Path


class LockError(RuntimeError):
    pass


class AppLock:
    """Single application lock released by the OS when the process exits."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+b")
        try:
            _lock_file(self._file)
        except OSError as exc:
            self._file.close()
            self._file = None
            raise LockError(f"Another GGE Assistant run is already active: {self.path}") from exc
        self._file.seek(0)
        self._file.truncate()
        self._file.write(str(os.getpid()).encode("ascii"))
        self._file.flush()

    def release(self) -> None:
        if self._file is None:
            return
        try:
            _unlock_file(self._file)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "AppLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


if os.name == "nt":
    import msvcrt

    def _lock_file(file_obj) -> None:
        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock_file(file_obj) -> None:
        file_obj.seek(0)
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _lock_file(file_obj) -> None:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(file_obj) -> None:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
