"""Run model-written code in a separate Python process and recover its circuit.

This is process isolation, not a security sandbox: the code runs with your
user's permissions. It gets a fresh temp directory, a timeout, and an
environment with API keys and tokens removed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from qiskit import QuantumCircuit, qpy

RUNNER = Path(__file__).with_name("_runner.py")
DEFAULT_TIMEOUT = 120.0
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass
class RunResult:
    ok: bool
    error: str | None
    stdout: str
    circuit: QuantumCircuit | None
    seconds: float


def _child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in _SECRET_HINTS)}
    env["MPLBACKEND"] = "Agg"  # never open plot windows
    return env


def run_code(code: str, timeout: float = DEFAULT_TIMEOUT) -> RunResult:
    """Execute `code` in a child interpreter and return the `qc` it defines."""
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ketwright-", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        (workdir / "candidate.py").write_text(code, encoding="utf-8")
        cmd = [sys.executable, "-I", "-X", "utf8", str(RUNNER), str(workdir)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=_child_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, f"The script did not finish within {timeout:.0f} seconds.", "", None,
                             time.perf_counter() - start)

        result_file = workdir / "result.json"
        if not result_file.exists():
            detail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            return RunResult(False, f"The Python process crashed (exit code {proc.returncode}).\n{detail}", "",
                             None, time.perf_counter() - start)

        result = json.loads(result_file.read_text(encoding="utf-8"))
        circuit = None
        if result.get("ok"):
            with open(workdir / "circuit.qpy", "rb") as fh:
                circuit = qpy.load(fh)[0]
        return RunResult(
            ok=bool(result.get("ok")),
            error=result.get("error"),
            stdout=result.get("stdout", ""),
            circuit=circuit,
            seconds=time.perf_counter() - start,
        )
