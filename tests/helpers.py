"""Test doubles: a scripted chat backend and an in-process code runner. No network, no API calls."""

from __future__ import annotations

from ketwright.backends import Reply, Usage
from ketwright.sandbox import RunResult


def fence(code: str) -> str:
    return f"Here is the script.\n\n```python\n{code.strip()}\n```\n"


GHZ3 = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
"""

# Uses an API that Qiskit 2.x removed, so it fails at import time.
GHZ3_OLD_API = """
from qiskit import QuantumCircuit, execute
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
"""

BASIS_1010_WRONG_ORDER = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.x(0)
qc.x(2)
"""

BASIS_1010_RIGHT = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.x(1)
qc.x(3)
"""

DONE = "VERDICT: DONE"


def inprocess_run(code: str) -> RunResult:
    """A fast stand-in for sandbox.run_code that executes in this process."""
    namespace: dict = {}
    try:
        exec(compile(code, "candidate.py", "exec"), namespace)
    except Exception as exc:  # noqa: BLE001
        return RunResult(False, f"{type(exc).__name__}: {exc}", "", None, 0.0)
    qc = namespace.get("qc")
    if qc is None:
        return RunResult(False, "The script did not define `qc`.", "", None, 0.0)
    return RunResult(True, None, "", qc, 0.0)


class ScriptedConversation:
    def __init__(self, system: str, replies: list):
        self.system = system
        self.replies = list(replies)
        self.received: list[str] = []

    def send(self, text: str) -> Reply:
        self.received.append(text)
        if not self.replies:
            raise AssertionError("the scripted backend ran out of replies")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, Reply):
            return item
        return Reply(item, Usage(input_tokens=100, output_tokens=50, seconds=0.5, calls=1), "stop")


class ScriptedBackend:
    """Each new conversation takes the next list of canned replies."""

    name = "scripted"

    def __init__(self, *scripts: list, model: str = "script-1"):
        self.model = model
        self.scripts = [list(s) for s in scripts]
        self.conversations: list[ScriptedConversation] = []

    def new_conversation(self, system: str) -> ScriptedConversation:
        if not self.scripts:
            raise AssertionError("no scripted conversation left")
        conv = ScriptedConversation(system, self.scripts.pop(0))
        self.conversations.append(conv)
        return conv
