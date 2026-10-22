"""Tests for tools._script_bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

from tools._script_bootstrap import ensure_repository_root_on_sys_path


def test_ensure_repository_root_on_sys_path(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    repo_root = tmp_path
    (tools_dir / "_repo_path.py").write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "_root = str(Path(__file__).resolve().parent.parent)",
                "if _root not in sys.path:",
                "    sys.path.insert(0, _root)",
            ]
        ),
        encoding="utf-8",
    )
    fake_script = tools_dir / "fake_tool.py"
    fake_script.write_text("", encoding="utf-8")

    returned = ensure_repository_root_on_sys_path(fake_script)

    assert returned == repo_root.resolve()
    assert str(repo_root.resolve()) in sys.path
