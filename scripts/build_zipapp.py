#!/usr/bin/env python3
from __future__ import annotations

import shutil
import stat
import tempfile
import zipapp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
TARGET = DIST / "tunnel.pyz"


def main() -> int:
    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tunnel-build-") as temp_dir:
        build_root = Path(temp_dir)
        shutil.copytree(
            ROOT / "tunnel",
            build_root / "tunnel",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (build_root / "__main__.py").write_text(
            "from tunnel.cli import main\nraise SystemExit(main())\n",
            encoding="utf-8",
        )
        zipapp.create_archive(
            build_root,
            target=TARGET,
            interpreter="/usr/bin/env python3",
            compressed=True,
        )
    current_mode = TARGET.stat().st_mode
    TARGET.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Built {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
