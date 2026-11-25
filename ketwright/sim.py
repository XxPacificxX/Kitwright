"""Simulation helpers shared by the agent's observations and the hidden checks."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlFlowOp
from qiskit.quantum_info import Operator, Statevector

# Instructions that carry no quantum action and can be dropped before
# building a statevector or an operator.
_IGNORABLE = {"barrier", "delay"}

# Largest circuit we will simulate exactly (statevector / operator).
MAX_EXACT_QUBITS = 12


class NotUnitaryError(ValueError):
    """The circuit contains something that has no statevector/operator form."""


def coherent_part(qc: QuantumCircuit, allow_initialize: bool) -> QuantumCircuit:
    """Return `qc` without final measurements, barriers and delays.

    Raises NotUnitaryError if what remains still has measurements, resets or
    classical control flow (or `initialize`, unless `allow_initialize`).
    """
    stripped = qc.remove_final_measurements(inplace=False)
    out = stripped.copy_empty_like()
    for inst in stripped.data:
        op = inst.operation
        if op.name in _IGNORABLE:
            continue
        if isinstance(op, ControlFlowOp):
            raise NotUnitaryError(f"classical control flow ({op.name}) is present")
        if op.name in ("measure", "reset"):
            raise NotUnitaryError(f"a mid-circuit '{op.name}' is present")
        if op.name == "initialize" and not allow_initialize:
            raise NotUnitaryError("'initialize' contains resets, so it is not a unitary")
        out.append(inst)
    return out


def final_statevector(qc: QuantumCircuit) -> Statevector:
    """Statevector produced from |0...0> (final measurements ignored)."""
    _check_size(qc)
    _check_bound(qc)
    return Statevector(coherent_part(qc, allow_initialize=True))


def unitary_of(qc: QuantumCircuit) -> Operator:
    """Operator of the circuit (final measurements ignored)."""
    _check_size(qc)
    _check_bound(qc)
    return Operator(coherent_part(qc, allow_initialize=False))


def sample_counts(qc: QuantumCircuit, shots: int, seed: int) -> dict[str, int]:
    """Sample measurement counts with Aer. Adds measure_all if nothing is measured.

    Keys are Qiskit bitstrings (highest classical bit on the left) with register
    separators removed.
    """
    from qiskit_aer import AerSimulator

    _check_bound(qc)
    if qc.num_qubits > 24:
        raise ValueError(f"{qc.num_qubits} qubits is too many to sample here")
    circuit = qc
    if not any(inst.operation.name == "measure" for inst in _all_instructions(qc)):
        circuit = qc.copy()
        circuit.measure_all()
    elif any(not qc.find_bit(bit).registers for bit in qc.clbits):
        # Aer misreports classical bits that belong to no register; put them all in one register.
        flat = QuantumCircuit(qc.num_qubits, qc.num_clbits, global_phase=qc.global_phase)
        circuit = flat.compose(qc, qubits=range(qc.num_qubits), clbits=range(qc.num_clbits))
    sim = AerSimulator(seed_simulator=seed)
    compiled = transpile(circuit, sim, seed_transpiler=seed)
    counts = sim.run(compiled, shots=shots).result().get_counts()
    merged: dict[str, int] = {}
    for key, value in counts.items():
        k = key.replace(" ", "")
        merged[k] = merged.get(k, 0) + value
    return merged


def marginal(counts: dict[str, int], clbits: list[int]) -> dict[str, int]:
    """Marginal counts over the given classical bit indices (highest index on the left)."""
    out: dict[str, int] = {}
    order = sorted(clbits, reverse=True)
    for key, value in counts.items():
        n = len(key)
        sub = "".join(key[n - 1 - i] if i < n else "0" for i in order)
        out[sub] = out.get(sub, 0) + value
    return out


def probabilities(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def ket(index: int, n: int) -> str:
    return "|" + format(index, f"0{n}b") + ">" if n else "|>"


def format_amplitude(a: complex) -> str:
    re, im = float(np.real(a)), float(np.imag(a))
    if abs(im) < 5e-5:
        return f"{re:+.4f}"
    if abs(re) < 5e-5:
        return f"{im:+.4f}j"
    return f"({re:+.4f}{im:+.4f}j)"


def has_measurements(qc: QuantumCircuit) -> bool:
    return any(inst.operation.name == "measure" for inst in _all_instructions(qc))


def _all_instructions(qc: QuantumCircuit):
    for inst in qc.data:
        yield inst
        if isinstance(inst.operation, ControlFlowOp):
            for block in inst.operation.blocks:
                yield from _all_instructions(block)


def _check_size(qc: QuantumCircuit) -> None:
    if qc.num_qubits > MAX_EXACT_QUBITS:
        raise ValueError(f"{qc.num_qubits} qubits is too many for exact simulation here")


def _check_bound(qc: QuantumCircuit) -> None:
    if qc.parameters:
        names = ", ".join(sorted(p.name for p in qc.parameters))
        raise ValueError(f"the circuit has unbound parameters: {names}")
