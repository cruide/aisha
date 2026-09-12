"""Tests for project-local debug log storage."""

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
