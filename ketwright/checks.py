"""Hidden evaluation checks. The agent never sees these.

Four kinds:
- statevector:   final state from |0...0> equals the reference state (fidelity, global phase ignored)
- unitary:       circuit unitary equals the reference unitary up to global phase,
                 optionally restricted to an allowed gate set
- reduced_state: the state of some qubits matches the reference (extra/ancilla qubits allowed)
- distribution:  sampled measurement outcomes match a target distribution or minimum probabilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.quantum_info import partial_trace, state_fidelity

from ketwright import sim

if TYPE_CHECKING:
    from ketwright.tasks import Task

CHECK_SHOTS = 4000
CHECK_SEED = 1234
STATE_TOL = 1e-6
DEFAULT_TVD_TOL = 0.05

ONE_QUBIT_GATES = frozenset(
    {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u", "u1", "u2", "u3", "r", "id"}
)
_ALWAYS_OK = frozenset({"barrier", "measure", "delay"})
_STANDARD = frozenset(get_standard_gate_name_mapping())


@dataclass
class CheckResult:
    passed: bool
    detail: str


def run_check(task: "Task", qc: QuantumCircuit | None) -> CheckResult:
    if qc is None:
        return CheckResult(False, "no circuit was produced")
    try:
        return _KINDS[task.kind](task, qc)
    except sim.NotUnitaryError as exc:
        return CheckResult(False, f"not a pure/unitary circuit: {exc}")
    except Exception as exc:  # noqa: BLE001 - any failure to evaluate is a failed check
        return CheckResult(False, f"{type(exc).__name__}: {exc}")


def disallowed_ops(qc: QuantumCircuit, allowed: frozenset[str] | set[str]) -> set[str]:
    """Gate names used outside `allowed`. Custom (non-library) gates are checked by their contents."""
    bad: set[str] = set()
    for inst in qc.data:
        op = inst.operation
        if op.name in allowed or op.name in _ALWAYS_OK:
            continue
        if isinstance(op, ControlFlowOp) or op.name in _STANDARD or op.definition is None:
            bad.add(op.name)
            continue
        bad |= disallowed_ops(op.definition, allowed)
    return bad


def _qubit_count(task: "Task", qc: QuantumCircuit, ref: QuantumCircuit) -> CheckResult | None:
    if qc.num_qubits != ref.num_qubits:
        return CheckResult(False, f"expected {ref.num_qubits} qubits, got {qc.num_qubits}")
    return None


def _check_statevector(task: "Task", qc: QuantumCircuit) -> CheckResult:
    ref = task.reference()
    if (bad := _qubit_count(task, qc, ref)) is not None:
        return bad
    got = sim.final_statevector(qc).data
    want = sim.final_statevector(ref).data
    fidelity = float(abs(np.vdot(want, got)) ** 2)
    return CheckResult(fidelity >= 1 - STATE_TOL, f"state fidelity {fidelity:.6f}")


def _check_unitary(task: "Task", qc: QuantumCircuit) -> CheckResult:
    ref = task.reference()
    if (bad := _qubit_count(task, qc, ref)) is not None:
        return bad
    allowed = task.params.get("allowed_ops")
    if allowed:
        used = disallowed_ops(qc, frozenset(allowed))
        if used:
            return CheckResult(False, f"uses gates outside the allowed set {sorted(allowed)}: {sorted(used)}")
    got = sim.unitary_of(qc).data
    want = sim.unitary_of(ref).data
    overlap = float(abs(np.trace(want.conj().T @ got)) / want.shape[0])
    return CheckResult(overlap >= 1 - STATE_TOL, f"|Tr(U_ref^dag U)|/d = {overlap:.6f}")


def _check_reduced_state(task: "Task", qc: QuantumCircuit) -> CheckResult:
    keep = list(task.params["qubits"])
    if qc.num_qubits <= max(keep):
        return CheckResult(False, f"circuit has {qc.num_qubits} qubits; qubit {max(keep)} is needed")
    ref = task.reference()

    def reduced(circuit: QuantumCircuit):
        state = sim.final_statevector(circuit)
        trace_out = [q for q in range(circuit.num_qubits) if q not in keep]
        return partial_trace(state, trace_out) if trace_out else state

    fidelity = float(state_fidelity(reduced(ref), reduced(qc), validate=False))
    return CheckResult(fidelity >= 1 - 1e-4, f"fidelity of qubits {keep}: {fidelity:.6f}")


def _check_distribution(task: "Task", qc: QuantumCircuit) -> CheckResult:
    params = task.params
    counts = sim.sample_counts(qc, shots=CHECK_SHOTS, seed=CHECK_SEED)
    if "clbits" in params:
        if qc.num_clbits <= max(params["clbits"]):
            return CheckResult(False, f"needs classical bit {max(params['clbits'])}, circuit has {qc.num_clbits}")
        counts = sim.marginal(counts, params["clbits"])
    probs = sim.probabilities(counts)
    expected = params.get("target") or params.get("min_prob")
    width = len(next(iter(expected)))
    widths = {len(k) for k in probs}
    if widths != {width}:
        return CheckResult(False, f"measured {sorted(widths)} classical bits per shot, expected {width}")

    top = ", ".join(f"'{k}': {v:.3f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:4])
    if "min_prob" in params:
        for outcome, need in params["min_prob"].items():
            got = probs.get(outcome, 0.0)
            if got < need:
                return CheckResult(False, f"P('{outcome}') = {got:.3f} < {need} (top: {top})")
        return CheckResult(True, f"top outcomes: {top}")
    tol = params.get("tol", DEFAULT_TVD_TOL)
    tvd = sim.total_variation(probs, params["target"])
    return CheckResult(tvd <= tol, f"total variation distance {tvd:.3f} (tol {tol}); top: {top}")


_KINDS = {
    "statevector": _check_statevector,
    "unitary": _check_unitary,
    "reduced_state": _check_reduced_state,
    "distribution": _check_distribution,
}
