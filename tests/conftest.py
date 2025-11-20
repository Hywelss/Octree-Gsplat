"""Pytest auto-configuration helpers."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _run_install(cmd: Sequence[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[tests] PyTorch auto-install command failed: {exc}")
        return False


def _ensure_torch_available() -> None:
    try:
        importlib.import_module("torch")
        return
    except ImportError:
        pass

    wheel_path = os.environ.get("PYTORCH_LOCAL_WHEEL")
    if wheel_path:
        candidate = Path(wheel_path).expanduser()
        if candidate.exists() and _run_install([sys.executable, "-m", "pip", "install", str(candidate)]):
            return

    version = os.environ.get("PYTORCH_CPU_VERSION", "2.1.2")
    index_url = os.environ.get("PYTORCH_CPU_INDEX_URL", "https://download.pytorch.org/whl/cpu")
    _run_install(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"torch=={version}",
            "--index-url",
            index_url,
        ]
    )


_ensure_torch_available()
