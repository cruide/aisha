"""Tests for project-local debug log storage."""

import logging
from pathlib import Path

import pytest

from aisha.logger import DebugLogger


@pytest.mark.parametrize("existing", [False, True])
def test_start_uses_project_log_directory(tmp_path: Path, existing: bool) -> None:
    """Create logs inside .aisha/logs, including when it already exists."""
    log_dir = tmp_path / ".aisha" / "logs"
    if existing:
        log_dir.mkdir(parents=True)
    logger = DebugLogger()
    try:
        path = logger.start(tmp_path)
        assert path.parent == log_dir
        assert logger.path == path
        assert path.is_file()
        assert "AISHA DEBUG LOG" in path.read_text(encoding="utf-8")
        assert not (tmp_path / "logs").exists()
    finally:
        logger.close()


def test_close_continues_after_handler_failure(tmp_path: Path) -> None:
    """Remove every handler even when one handler cannot be closed."""
    class BrokenHandler(logging.Handler):
        failed = False

        def close(self) -> None:
            if not self.failed:
                self.failed = True
                raise OSError("broken")
            super().close()

    class RecordingHandler(logging.Handler):
        closed = False

        def close(self) -> None:
            self.closed = True

    logger = DebugLogger()
    logger.start(tmp_path)
    broken = BrokenHandler()
    recording = RecordingHandler()
    assert logger._logger is not None
    logger._logger.addHandler(broken)
    logger._logger.addHandler(recording)

    logger.close()

    assert recording.closed is True
    assert logger._logger is None
    logger.close()
