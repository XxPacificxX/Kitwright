"""Turn a run of model-written code into a plain-text execution report.

The report is what the agent sees when it checks its own work. It describes
what the circuit actually does (statevector, basis-state action, sampled
counts); it never uses the hidden evaluation checks.
"""

from __future__ import annotations

import numpy as np

from ketwright import sim
from ketwright.sandbox import RunResult

OBSERVATION_SHOTS = 2000
OBSERVATION_SEED = 7  # differs from the checker's seed on purpose
MAX_TERMS = 32
MAX_ACTION_TERMS = 80


def describe(run: RunResult) -> str:
    if not run.ok or run.circuit is None:
        parts = ["The script FAILED.", run.error or "Unknown error."]
        if run.stdout.strip():
            parts.append("Output before the failure:\n" + run.stdout.strip()[-1000:])
        return "\n".join(parts)

    qc = run.circuit
    ops = ", ".join(f"{name}: {count}" for name, count in qc.count_ops().items()) or "none"
    lines = [
        "The script ran and defined `qc`.",
        f"Circuit: {qc.num_qubits} qubits, {qc.num_clbits} classical bits, depth {qc.depth()}.",
        f"Top-level operations: {ops}.",
    ]
    if qc.parameters:
        lines.append("Unbound parameters: " + ", ".join(sorted(p.name for p in qc.parameters)) + ".")
        return "\n".join(lines)

    lines.extend(_state_section(qc))
    lines.extend(_action_section(qc))
    if sim.has_measurements(qc):
        lines.extend(_counts_section(qc))
    if run.stdout.strip():
        lines.append("Script output (tail):\n" + run.stdout.strip()[-800:])
    return "\n".join(lines)


def _state_section(qc) -> list[str]:
    try:
        state = sim.final_statevector(qc)
    except Exception as exc:  # noqa: BLE001
        return [f"Final statevector: not shown ({exc})."]
    data = state.data
    order = np.argsort(-np.abs(data))
    terms = [i for i in order if abs(data[i]) > 1e-6][:MAX_TERMS]
    n = qc.num_qubits
    header = "Final statevector from |0...0> (final measurements ignored; kets are |q{}...q0>):".format(n - 1)
    rows = [f"  {sim.format_amplitude(data[i])} {sim.ket(int(i), n)}  (p={abs(data[i]) ** 2:.4f})"
            for i in sorted(terms)]
    hidden = int(np.sum(np.abs(data) > 1e-6)) - len(terms)
    if hidden > 0:
        rows.append(f"  ... and {hidden} smaller terms")
    return [header, *rows]


def _action_section(qc) -> list[str]:
    """Show U|j> for every basis state j when the circuit is small and unitary."""
    if qc.num_qubits > 3:
        return []
    try:
        matrix = sim.unitary_of(qc).data
    except Exception:  # noqa: BLE001 - not unitary: nothing to show
        return []
    n = qc.num_qubits
    nonzero = int(np.sum(np.abs(matrix) > 1e-6))
    if nonzero > MAX_ACTION_TERMS:
        return []
    rows = ["Action on each basis state (the circuit's unitary, column by column):"]
    for j in range(2**n):
        column = matrix[:, j]
        terms = " ".join(f"{sim.format_amplitude(column[k])}{sim.ket(k, n)}"
                         for k in range(2**n) if abs(column[k]) > 1e-6)
        rows.append(f"  {sim.ket(j, n)} -> {terms}")
    return rows


def _counts_section(qc) -> list[str]:
    try:
        counts = sim.sample_counts(qc, shots=OBSERVATION_SHOTS, seed=OBSERVATION_SEED)
    except Exception as exc:  # noqa: BLE001
        return [f"Measurement counts: simulation failed ({type(exc).__name__}: {exc})."]
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:16]
    rows = [f"Measurement counts from {OBSERVATION_SHOTS} shots (keys: highest classical bit on the left):"]
    rows += [f"  '{key}': {value}" for key, value in top]
    if len(counts) > len(top):
        rows.append(f"  ... and {len(counts) - len(top)} rarer outcomes")
    return rows
