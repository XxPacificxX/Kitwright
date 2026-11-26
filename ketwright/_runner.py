"""Child-process entry point: run one generated script and save its `qc`.

Usage: python -I -X utf8 _runner.py <workdir>

Reads <workdir>/candidate.py, executes it, and writes <workdir>/result.json
(plus <workdir>/circuit.qpy when a QuantumCircuit named `qc` was produced).
This file is run as a standalone script, so it imports nothing from ketwright.
"""

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

MAX_OUTPUT = 4000


def _short_traceback() -> str:
    text = traceback.format_exc()
    # Drop the runner's own frames so the model sees only its code.
    marker = 'File "candidate.py"'
    first = text.find(marker)
    if first != -1:
        text = "Traceback (most recent call last):\n  " + text[first:]
    # Library paths sit under the user's home folder; keep usernames out of reports and logs.
    text = text.replace(str(Path.home()), "~")
    return text[-3000:]


def main() -> int:
    workdir = Path(sys.argv[1])
    source = (workdir / "candidate.py").read_text(encoding="utf-8")
    result = {"ok": False, "error": None, "stdout": ""}
    namespace = {"__name__": "__main__", "__file__": "candidate.py"}
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compile(source, "candidate.py", "exec"), namespace)
    except BaseException:  # noqa: BLE001 - report SystemExit/KeyboardInterrupt too
        result["error"] = _short_traceback()
    else:
        from qiskit import QuantumCircuit, qpy

        qc = namespace.get("qc")
        if qc is None:
            result["error"] = "The script ran, but it did not define a top-level variable named `qc`."
        elif not isinstance(qc, QuantumCircuit):
            result["error"] = f"`qc` must be a qiskit.QuantumCircuit, but it is a {type(qc).__name__}."
        else:
            try:
                with open(workdir / "circuit.qpy", "wb") as fh:
                    qpy.dump(qc, fh)
                result["ok"] = True
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"The circuit could not be serialized with QPY: {type(exc).__name__}: {exc}"
    result["stdout"] = buffer.getvalue()[-MAX_OUTPUT:]
    (workdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
