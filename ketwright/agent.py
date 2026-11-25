"""The Ketwright agent: write Qiskit code, run it, read the result, fix it.

The loop only ever sees what the circuit *does* (errors, statevector,
basis-state action, sampled counts). It never sees the hidden evaluation
checks, so "self-correcting" here means correcting from its own reading of
the execution report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ketwright.backends import Backend, BackendError, Usage
from ketwright.observe import describe
from ketwright.sandbox import RunResult, run_code

SYSTEM_PROMPT = """You are Ketwright, an expert quantum software engineer. You turn plain-English quantum computing tasks into correct Qiskit code.

How to answer
- Reply with one ```python code block that holds a complete, self-contained script.
- The script must define a top-level variable `qc`: the final qiskit.QuantumCircuit.
- Target Qiskit 2.x. Do not run simulators, call IBM services, plot, read or write files, or use the network. The harness simulates `qc` for you and reports back.
- Bind every circuit parameter to a number.
- Add measurements only when the task asks for them.

Conventions (the tasks use them too)
- Qubit 0 is the least significant bit. A ket or bitstring such as |110> or '110' lists q2 q1 q0 from left to right, so '110' means q2=1, q1=1, q0=0, and the basis-state index is sum(q_i * 2**i).
- Measurement count keys list classical bits from the highest index (left) down to bit 0 (right).
- Global phase does not matter unless the task says it does.

Qiskit 2.x reminders
- `execute`, `qiskit.Aer`, `qiskit.algorithms`, `QuantumCircuit.cnot`/`toffoli`/`mct` and `c_if` no longer exist.
- Use qc.cx, qc.ccx, qc.mcx(controls, target), qc.cp, qc.rzz, and `with qc.if_test((clbit, 1)):` for classical feed-forward.
- Library pieces that work: QFTGate, StatePreparation, MCXGate, UnitaryGate and grover_operator() from qiskit.circuit.library."""

FIRST_PROMPT = "Task: {task}\n\nWrite the Qiskit script."

REVIEW_PROMPT = """Here is what happened when your script ran (attempt {n} of {max}):

{report}

Check this result against the task step by step: number of qubits, which qubit is which, the bit order of measured results, signs and phases, and probabilities.
- If the circuit does exactly what the task asks, reply with the single line: VERDICT: DONE
- Otherwise, say in one or two sentences what is wrong, then give the complete corrected script in one ```python block."""

FAILED_PROMPT = """Your script failed (attempt {n} of {max}):

{report}

Fix the problem and reply with the complete corrected script in one ```python block."""

NO_CODE_PROMPT = ("I could not find a ```python code block in your reply. Reply with the complete script "
                  "in one ```python block that defines `qc`.")

DONE_BUT_FAILED_PROMPT = ("The last script failed, so the task is not done yet. Reply with the complete "
                          "corrected script in one ```python block.")

_FENCE = re.compile(r"```[ \t]*(?:python3|python|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python3|python|py)?[ \t]*\r?\n", re.I)
_DEFINES_QC = re.compile(r"^qc\s*(?::[^=\n]+)?=", re.M)
_DONE = re.compile(r"VERDICT:\s*DONE", re.I)


def extract_code(text: str) -> str | None:
    """Return the script from a reply: the last fenced block that defines `qc`, else the last block."""
    blocks = [b.strip() for b in _FENCE.findall(text) if b.strip()]
    if not blocks:
        # A reply cut off by the token limit can have an opening fence and no closing one.
        opens = list(_OPEN_FENCE.finditer(text))
        if not opens:
            return None
        tail = text[opens[-1].end():].strip()
        return tail or None
    with_qc = [b for b in blocks if _DEFINES_QC.search(b)]
    return (with_qc or blocks)[-1]


def says_done(text: str) -> bool:
    return bool(_DONE.search(text))


@dataclass
class Attempt:
    number: int  # 1-based attempt (round) number
    call: int  # which model call produced this code (1 = the first reply)
    code: str
    run: RunResult
    report: str


@dataclass
class AgentResult:
    task: str
    attempts: list[Attempt]
    status: str  # verified | out_of_rounds | out_of_calls | refused | error
    usage: Usage
    first_call_usage: Usage
    error: str | None = None
    transcript: list[dict[str, str]] = field(default_factory=list)

    @property
    def final(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def one_shot(self) -> Attempt | None:
        """The code from the very first reply, i.e. what a no-feedback agent would return."""
        if self.attempts and self.attempts[0].call == 1:
            return self.attempts[0]
        return None


EventHandler = Callable[[str, dict[str, Any]], None]


def solve(
    task: str,
    backend: Backend,
    max_rounds: int = 4,
    run: Callable[[str], RunResult] = run_code,
    on_event: EventHandler | None = None,
) -> AgentResult:
    """Run the write -> execute -> review -> fix loop for one task.

    `max_rounds` caps the number of code attempts; `max_rounds=1` is plain one-shot.
    Raises BackendError only for fatal backend problems (bad key, unknown model).
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    emit = on_event or (lambda kind, payload: None)
    conversation = backend.new_conversation(SYSTEM_PROMPT)
    message = FIRST_PROMPT.format(task=task)
    attempts: list[Attempt] = []
    transcript: list[dict[str, str]] = []
    usage = Usage()
    first_usage = Usage()
    status = "out_of_calls"
    error = None
    max_calls = 2 * max_rounds + 1

    for call in range(1, max_calls + 1):
        transcript.append({"role": "user", "text": message})
        try:
            reply = conversation.send(message)
        except BackendError as exc:
            if exc.fatal:
                raise
            status, error = "error", str(exc)
            break
        usage = usage + reply.usage
        if call == 1:
            first_usage = reply.usage
        transcript.append({"role": "assistant", "text": reply.text})
        emit("reply", {"call": call, "text": reply.text, "usage": reply.usage})

        if reply.stop_reason == "refusal":
            status = "refused"
            break

        code = extract_code(reply.text)
        last = attempts[-1] if attempts else None
        repeated = last is not None and code is not None and code.strip() == last.code.strip()
        if code is None or (repeated and says_done(reply.text)):
            if last is not None and says_done(reply.text):
                if last.run.ok:
                    status = "verified"
                    break
                message = DONE_BUT_FAILED_PROMPT
            else:
                message = NO_CODE_PROMPT
            continue

        result = run(code)
        report = describe(result)
        attempt = Attempt(len(attempts) + 1, call, code, result, report)
        attempts.append(attempt)
        emit("attempt", {"attempt": attempt})

        if len(attempts) >= max_rounds:
            status = "out_of_rounds"
            break
        template = REVIEW_PROMPT if result.ok else FAILED_PROMPT
        message = template.format(n=attempt.number, max=max_rounds, report=report)

    return AgentResult(task, attempts, status, usage, first_usage, error, transcript)
