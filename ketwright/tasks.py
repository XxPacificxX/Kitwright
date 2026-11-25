"""The evaluation set: 50 plain-English quantum tasks, each with a reference and a hidden check.

Every reference circuit is validated against its own check in the test suite,
and the checks are also tested against deliberately wrong circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, StatePreparation, UnitaryGate

from ketwright.checks import ONE_QUBIT_GATES


@dataclass
class Task:
    id: str
    category: str
    prompt: str
    kind: str  # statevector | unitary | reduced_state | distribution
    reference: Callable[[], QuantumCircuit]
    params: dict[str, Any] = field(default_factory=dict)
    # A known-good answer that also respects any gate restrictions; defaults to the reference.
    solution: Callable[[], QuantumCircuit] | None = None

    def example(self) -> QuantumCircuit:
        return (self.solution or self.reference)()


# --------------------------------------------------------------------------- helpers

def _state(amplitudes) -> QuantumCircuit:
    vec = np.asarray(amplitudes, dtype=complex)
    vec = vec / np.linalg.norm(vec)
    n = int(round(np.log2(len(vec))))
    qc = QuantumCircuit(n)
    qc.append(StatePreparation(vec), range(n))
    return qc


def _basis_superposition(n: int, indices) -> QuantumCircuit:
    amps = np.zeros(2**n)
    amps[list(indices)] = 1
    return _state(amps)


def _unitary(matrix) -> QuantumCircuit:
    matrix = np.asarray(matrix, dtype=complex)
    n = int(round(np.log2(matrix.shape[0])))
    qc = QuantumCircuit(n)
    qc.append(UnitaryGate(matrix), range(n))
    return qc


def dft_matrix(n: int, inverse: bool = False) -> np.ndarray:
    size = 2**n
    sign = -1 if inverse else 1
    j, k = np.meshgrid(np.arange(size), np.arange(size))  # column j (input), row k (output)
    return np.exp(sign * 2j * pi * j * k / size) / np.sqrt(size)


def _permutation(n: int, f: Callable[[int], int]) -> np.ndarray:
    size = 2**n
    m = np.zeros((size, size))
    for j in range(size):
        m[f(j), j] = 1
    return m


def _phase_oracle(n: int, marked: str) -> QuantumCircuit:
    """Flip the sign of basis state `marked` (Qiskit order: q_{n-1} ... q_0)."""
    oracle = QuantumCircuit(n)
    zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
    if zeros:
        oracle.x(zeros)
    oracle.h(n - 1)
    oracle.mcx(list(range(n - 1)), n - 1)
    oracle.h(n - 1)
    if zeros:
        oracle.x(zeros)
    return oracle


def _diffuser(n: int) -> QuantumCircuit:
    d = QuantumCircuit(n)
    d.h(range(n))
    d.x(range(n))
    d.h(n - 1)
    d.mcx(list(range(n - 1)), n - 1)
    d.h(n - 1)
    d.x(range(n))
    d.h(range(n))
    return d


def _grover(n: int, marked: str, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(_phase_oracle(n, marked), inplace=True)
        qc.compose(_diffuser(n), inplace=True)
    qc.measure(range(n), range(n))
    return qc


def _qpe(angle: float, counting: int = 3) -> QuantumCircuit:
    """QPE of the phase gate P(angle) on eigenstate |1>; counting qubits 0..t-1, target qubit t."""
    qc = QuantumCircuit(counting + 1, counting)
    qc.x(counting)
    qc.h(range(counting))
    for q in range(counting):
        qc.cp(angle * 2**q, q, counting)
    qc.append(QFTGate(counting).inverse(), range(counting))
    qc.measure(range(counting), range(counting))
    return qc


def _circuit(n: int, build: Callable[[QuantumCircuit], None], clbits: int = 0) -> Callable[[], QuantumCircuit]:
    def make() -> QuantumCircuit:
        qc = QuantumCircuit(n, clbits) if clbits else QuantumCircuit(n)
        build(qc)
        return qc

    return make


# --------------------------------------------------------------------------- reference builders

def _ghz(n: int):
    def build(qc):
        qc.h(0)
        for q in range(n - 1):
            qc.cx(q, q + 1)

    return _circuit(n, build)


def _cluster3(qc):
    qc.h(range(3))
    qc.cz(0, 1)
    qc.cz(1, 2)


def _split_bell(qc):
    qc.h(0)
    qc.cx(0, 3)
    qc.x([1, 2])


def _product_mixed(qc):
    qc.h(0)
    qc.x(1)
    qc.h(1)
    qc.x(2)


def _bloch(qc):
    qc.ry(pi / 3, 0)
    qc.p(pi / 4, 0)


def _plus_i(qc):
    qc.h(0)
    qc.s(0)


def _cyclic_shift(qc):
    qc.swap(1, 2)
    qc.swap(0, 1)


def _swap_cx(qc):
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)


def _toffoli_ct(qc):
    qc.h(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(1)
    qc.t(2)
    qc.h(2)
    qc.cx(0, 1)
    qc.t(0)
    qc.tdg(1)
    qc.cx(0, 1)


def _cz_h_cx(qc):
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)


def _ch_ry_cx(qc):
    qc.ry(pi / 4, 1)
    qc.cx(0, 1)
    qc.ry(-pi / 4, 1)


def _iswap_cx(qc):
    qc.s([0, 1])
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.h(1)


def _crx_decomposed(qc):
    theta = pi / 3
    qc.rz(pi / 2, 1)
    qc.cx(0, 1)
    qc.ry(-theta / 2, 1)
    qc.cx(0, 1)
    qc.ry(theta / 2, 1)
    qc.rz(-pi / 2, 1)


def _trotter(qc):
    qc.rz(0.2, 0)
    qc.rz(0.2, 1)
    qc.rxx(0.2, 0, 1)


def _bv_1011(qc):
    qc.x(4)
    qc.h(range(5))
    for q in (0, 1, 3):  # s = '1011' -> s0 = s1 = s3 = 1
        qc.cx(q, 4)
    qc.h(range(4))
    qc.measure(range(4), range(4))


def _dj_balanced(qc):
    qc.x(3)
    qc.h(range(4))
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.h(range(3))
    qc.measure(range(3), range(3))


def _teleport(qc):
    qc.ry(pi / 3, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(2)
    with qc.if_test((qc.clbits[0], 1)):
        qc.z(2)
    qc.measure(2, 2)


def _superdense_10(qc):
    qc.h(0)
    qc.cx(0, 1)
    qc.x(0)  # encodes '10' for this decoder
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])


def _qrng(qc):
    qc.h(range(3))
    qc.measure(range(3), range(3))


def _simon_11(qc):
    qc.h([0, 1])
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.h([0, 1])
    qc.measure([0, 1], [0, 1])


def _swap_test(qc):
    qc.h(2)  # |+> on qubit 2, |0> on qubit 1
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    qc.measure(0, 0)


def _half_adder(qc):
    qc.x([0, 1])
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.ccx(0, 1, 3)
    qc.measure(2, 0)
    qc.measure(3, 1)


def _bell_measurement(qc):
    qc.h(0)
    qc.cx(0, 1)
    qc.x(1)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])


def _qaoa_triangle(qc):
    gamma, beta = 0.4, 0.7
    qc.h(range(3))
    for a, b in ((0, 1), (1, 2), (0, 2)):
        qc.rzz(2 * gamma, a, b)
    qc.rx(2 * beta, range(3))


def _angle_encoding(qc):
    for q, x in enumerate((0.1, 0.5, 0.9)):
        qc.ry(x, q)


def _bitflip_recovery(qc):
    qc.ry(0.8, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.x(1)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.ccx(2, 1, 0)


def _ry_target(angle: float):
    return _circuit(1, lambda qc: qc.ry(angle, 0))


def _phaseflip_encode(qc):
    qc.ry(1.1, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(range(3))


# --------------------------------------------------------------------------- the task list

_BIT_ORDER = "Qiskit bit order: qubit 0 is the rightmost bit"
_DICKE_INDICES = [i for i in range(16) if bin(i).count("1") == 2]

TASKS: list[Task] = [
    # ---- state preparation (statevector checks) ----
    Task("sp01_bell_phi_plus", "state-prep", "Prepare the Bell state (|00> + |11>)/sqrt(2) on 2 qubits.",
         "statevector", _ghz(2)),
    Task("sp02_bell_psi_minus", "state-prep",
         f"Prepare the Bell state (|01> - |10>)/sqrt(2) on 2 qubits ({_BIT_ORDER}).",
         "statevector", lambda: _state([0, 1, -1, 0])),
    Task("sp03_ghz3", "state-prep", "Prepare a 3-qubit GHZ state (|000> + |111>)/sqrt(2).",
         "statevector", _ghz(3)),
    Task("sp04_ghz5", "state-prep", "Prepare a 5-qubit GHZ state.", "statevector", _ghz(5)),
    Task("sp05_w3", "state-prep", "Prepare the 3-qubit W state (|001> + |010> + |100>)/sqrt(3).",
         "statevector", lambda: _basis_superposition(3, [1, 2, 4])),
    Task("sp06_uniform4", "state-prep",
         "Prepare the equal superposition of all 16 computational basis states of 4 qubits.",
         "statevector", _circuit(4, lambda qc: qc.h(range(4)))),
    Task("sp07_basis_1010", "state-prep",
         f"Prepare the 4-qubit computational basis state |1010> ({_BIT_ORDER}, qubit 3 the leftmost).",
         "statevector", _circuit(4, lambda qc: qc.x([1, 3]))),
    Task("sp08_plus_i", "state-prep", "Prepare the single-qubit state (|0> + i|1>)/sqrt(2).",
         "statevector", _circuit(1, _plus_i)),
    Task("sp09_bloch", "state-prep",
         "Prepare the single-qubit state with Bloch-sphere angles theta = pi/3 and phi = pi/4, "
         "that is cos(theta/2)|0> + e^(i*phi) sin(theta/2)|1>.",
         "statevector", _circuit(1, _bloch)),
    Task("sp10_dicke42", "state-prep",
         "Prepare the 4-qubit Dicke state with two excitations: the equal superposition of the 6 "
         "computational basis states that contain exactly two 1s.",
         "statevector", lambda: _basis_superposition(4, _DICKE_INDICES)),
    Task("sp11_ghz_minus", "state-prep", "Prepare the 3-qubit state (|000> - |111>)/sqrt(2).",
         "statevector", lambda: _state([1, 0, 0, 0, 0, 0, 0, -1])),
    Task("sp12_cluster3", "state-prep",
         "Prepare the 3-qubit linear cluster state, i.e. the graph state of the path graph 0 - 1 - 2.",
         "statevector", _circuit(3, _cluster3)),
    Task("sp13_uniform_0_to_4", "state-prep",
         "On 3 qubits, prepare the equal superposition of the basis states whose integer values are "
         "0, 1, 2, 3 and 4 (integer value = sum of q_i * 2^i), with no other basis states present.",
         "statevector", lambda: _basis_superposition(3, range(5))),
    Task("sp14_split_bell", "state-prep",
         "On 4 qubits, put qubits 0 and 3 into the Bell state (|00> + |11>)/sqrt(2) and set qubits 1 "
         "and 2 both to |1>.",
         "statevector", _circuit(4, _split_bell)),
    Task("sp15_product_mixed", "state-prep",
         "On 3 qubits, prepare qubit 0 in |+>, qubit 1 in |->, and qubit 2 in |1>.",
         "statevector", _circuit(3, _product_mixed)),

    # ---- unitaries and gate synthesis (unitary checks) ----
    Task("un01_qft3", "unitary",
         "Build a 3-qubit quantum Fourier transform: the unitary that maps |j> to "
         "(1/sqrt(8)) * sum_k exp(2*pi*i*j*k/8) |k>, where j and k are basis-state integers in "
         "Qiskit's little-endian order. Include any swaps needed for this to hold exactly.",
         "unitary", lambda: _unitary(dft_matrix(3))),
    Task("un02_iqft4", "unitary",
         "Build the 4-qubit inverse quantum Fourier transform: the unitary that maps |j> to "
         "(1/4) * sum_k exp(-2*pi*i*j*k/16) |k>, with j and k basis-state integers in Qiskit's "
         "little-endian order. Include any swaps needed for this to hold exactly.",
         "unitary", lambda: _unitary(dft_matrix(4, inverse=True))),
    Task("un03_swap_from_cx", "unitary", "Implement a SWAP of qubits 0 and 1 using only CNOT (cx) gates.",
         "unitary", _circuit(2, lambda qc: qc.swap(0, 1)), {"allowed_ops": {"cx"}}, _circuit(2, _swap_cx)),
    Task("un04_toffoli_clifford_t", "unitary",
         "Implement the Toffoli gate (controls qubits 0 and 1, target qubit 2) using only H, T, "
         "T-dagger and CNOT gates.",
         "unitary", _circuit(3, lambda qc: qc.ccx(0, 1, 2)), {"allowed_ops": {"h", "t", "tdg", "cx"}},
         _circuit(3, _toffoli_ct)),
    Task("un05_cz_from_h_cx", "unitary", "Implement a CZ gate on qubits 0 and 1 using only H and CNOT gates.",
         "unitary", _circuit(2, lambda qc: qc.cz(0, 1)), {"allowed_ops": {"h", "cx"}}, _circuit(2, _cz_h_cx)),
    Task("un06_ch_from_ry_cx", "unitary",
         "Implement a controlled-Hadamard gate (control qubit 0, target qubit 1) using only RY and CNOT "
         "gates. It must equal the controlled-H exactly, up to a global phase.",
         "unitary", _circuit(2, lambda qc: qc.ch(0, 1)), {"allowed_ops": {"ry", "cx"}}, _circuit(2, _ch_ry_cx)),
    Task("un07_iswap", "unitary",
         "Implement the iSWAP gate on qubits 0 and 1 (|01> -> i|10>, |10> -> i|01>, |00> and |11> "
         "unchanged) using only CNOT gates and single-qubit gates.",
         "unitary", _circuit(2, lambda qc: qc.iswap(0, 1)), {"allowed_ops": set(ONE_QUBIT_GATES) | {"cx"}},
         _circuit(2, _iswap_cx)),
    Task("un08_controlled_s", "unitary",
         "Implement a controlled-S gate with control qubit 0 and target qubit 1: it multiplies |11> by i "
         "and leaves the other basis states unchanged.",
         "unitary", _circuit(2, lambda qc: qc.cs(0, 1))),
    Task("un09_increment3", "unitary",
         "Build a 3-qubit increment circuit that maps |x> to |x + 1 mod 8>, where x is the basis-state "
         "integer in Qiskit's little-endian order.",
         "unitary", lambda: _unitary(_permutation(3, lambda j: (j + 1) % 8))),
    Task("un10_cyclic_shift3", "unitary",
         "Build a 3-qubit circuit that cyclically moves qubit states: the state of qubit 0 ends up on "
         "qubit 1, the state of qubit 1 on qubit 2, and the state of qubit 2 on qubit 0.",
         "unitary", _circuit(3, _cyclic_shift)),
    Task("un11_phase_oracle_110", "unitary",
         "Build a 3-qubit phase oracle that multiplies the basis state |110> (q2 = 1, q1 = 1, q0 = 0) by "
         "-1 and leaves every other basis state unchanged.",
         "unitary", lambda: _unitary(np.diag([1, 1, 1, 1, 1, 1, -1, 1]))),
    Task("un12_diffusion3", "unitary",
         "Build the 3-qubit Grover diffusion operator 2|s><s| - I, where |s> is the uniform superposition "
         "of all 8 basis states (a global phase difference is fine).",
         "unitary", lambda: _unitary(2 / 8 * np.ones((8, 8)) - np.eye(8))),
    Task("un13_crx_decomposed", "unitary",
         "Implement a controlled-RX(pi/3) gate (control qubit 0, target qubit 1) using only RX, RY, RZ and "
         "CNOT gates.",
         "unitary", _circuit(2, lambda qc: qc.crx(pi / 3, 0, 1)), {"allowed_ops": {"rx", "ry", "rz", "cx"}},
         _circuit(2, _crx_decomposed)),
    Task("un14_fredkin", "unitary",
         "Implement a Fredkin (controlled-SWAP) gate: control qubit 0, swapping qubits 1 and 2.",
         "unitary", _circuit(3, lambda qc: qc.cswap(0, 1, 2))),
    Task("un15_zz_evolution", "unitary",
         "Implement the two-qubit time evolution U = exp(-i * 0.3 * Z0 Z1), where Z0 Z1 is the tensor "
         "product of Pauli Z on qubits 0 and 1.",
         "unitary", _circuit(2, lambda qc: qc.rzz(0.6, 0, 1))),
    Task("un16_trotter_step", "unitary",
         "Implement one Trotter step on 2 qubits: first apply exp(-i * 0.1 * (Z0 + Z1)), then apply "
         "exp(-i * 0.1 * X0 X1), where Z0 is Pauli Z on qubit 0 and X0 X1 is Pauli X on both qubits.",
         "unitary", _circuit(2, _trotter)),

    # ---- algorithms with measurement (distribution checks) ----
    Task("ms01_grover3_101", "algorithm",
         "Use Grover's search on 3 qubits to find the marked state |101>. Use the number of iterations "
         "that maximizes the success probability, then measure qubit i into classical bit i of a 3-bit "
         "register.",
         "distribution", lambda: _grover(3, "101", 2), {"min_prob": {"101": 0.9}}),
    Task("ms02_grover2_11", "algorithm",
         "Use Grover's search on 2 qubits to find the marked state |11>, using the optimal number of "
         "iterations, then measure qubit i into classical bit i.",
         "distribution", lambda: _grover(2, "11", 1), {"min_prob": {"11": 0.98}}),
    Task("ms03_grover4_1011", "algorithm",
         "Use Grover's search on 4 qubits to find the marked state |1011> (q3 = 1, q2 = 0, q1 = 1, "
         "q0 = 1). Use the number of iterations that maximizes the success probability, then measure "
         "qubit i into classical bit i of a 4-bit register.",
         "distribution", lambda: _grover(4, "1011", 3), {"min_prob": {"1011": 0.9}}),
    Task("ms04_bernstein_vazirani_1011", "algorithm",
         "Implement Bernstein-Vazirani for the hidden string s = '1011' (Qiskit order, so s3 = 1, s2 = 0, "
         "s1 = 1, s0 = 1) and f(x) = s . x mod 2. Use 4 input qubits (0-3) and one output qubit (4). "
         "Measure input qubit i into classical bit i of a 4-bit register so the result reads '1011'.",
         "distribution", _circuit(5, _bv_1011, clbits=4), {"target": {"1011": 1.0}, "tol": 0.02}),
    Task("ms05_deutsch_jozsa_balanced", "algorithm",
         "Run Deutsch-Jozsa on the balanced function f(x) = x0 XOR x1 of 3 input bits, with input qubits "
         "0-2 and output qubit 3. Measure input qubit i into classical bit i of a 3-bit register.",
         "distribution", _circuit(4, _dj_balanced, clbits=3), {"target": {"011": 1.0}, "tol": 0.02}),
    Task("ms06_teleportation", "algorithm",
         "Teleport the state RY(pi/3)|0> from qubit 0 to qubit 2, using a Bell pair on qubits 1 and 2. "
         "Use one 3-bit classical register: measure qubit 0 into bit 0 and qubit 1 into bit 1, apply the "
         "X and Z corrections to qubit 2 conditioned on those bits (classical feed-forward), then measure "
         "qubit 2 into bit 2.",
         "distribution", _circuit(3, _teleport, clbits=3),
         {"clbits": [2], "target": {"0": 0.75, "1": 0.25}, "tol": 0.04}),
    Task("ms07_superdense_coding", "algorithm",
         "Superdense coding: qubit 0 (Alice) and qubit 1 (Bob) start in (|00> + |11>)/sqrt(2). Alice "
         "encodes the 2-bit message '10' by acting on qubit 0 only, then Bob decodes and measures qubit i "
         "into classical bit i. The measured key (bit 1 on the left) must equal '10'.",
         "distribution", _circuit(2, _superdense_10, clbits=2), {"target": {"10": 1.0}, "tol": 0.02}),
    Task("ms08_qpe_t_gate", "algorithm",
         "Use quantum phase estimation with 3 counting qubits (qubits 0-2) to estimate the phase of the "
         "T gate on its eigenstate |1> (target qubit 3). T|1> = exp(2*pi*i*phi)|1> with phi = 1/8. "
         "Measure counting qubit i into classical bit i, so the 3-bit result read as an integer is "
         "8 * phi.",
         "distribution", lambda: _qpe(pi / 4), {"min_prob": {"001": 0.95}}),
    Task("ms09_qpe_phase_3_8", "algorithm",
         "Use quantum phase estimation with 3 counting qubits (qubits 0-2) to estimate phi for the phase "
         "gate P(3*pi/4), whose eigenstate |1> (target qubit 3) picks up exp(2*pi*i*phi). Measure "
         "counting qubit i into classical bit i, so the 3-bit result read as an integer is 8 * phi.",
         "distribution", lambda: _qpe(3 * pi / 4), {"min_prob": {"011": 0.95}}),
    Task("ms10_qrng3", "algorithm",
         "Build a 3-qubit quantum random number generator: measuring the 3 qubits (qubit i into "
         "classical bit i) gives each of the 8 outcomes with probability 1/8.",
         "distribution", _circuit(3, _qrng, clbits=3),
         {"target": {format(i, "03b"): 1 / 8 for i in range(8)}, "tol": 0.05}),
    Task("ms11_simon_11", "algorithm",
         "Run one round of Simon's algorithm for a 2-bit function with hidden period s = '11', so "
         "f(x) = f(x XOR 11) and f is two-to-one. Use input qubits 0-1 and output qubits 2-3, and measure "
         "input qubit i into classical bit i of a 2-bit register.",
         "distribution", _circuit(4, _simon_11, clbits=2),
         {"target": {"00": 0.5, "11": 0.5}, "tol": 0.05}),
    Task("ms12_swap_test", "algorithm",
         "Use a swap test to compare |0> on qubit 1 with |+> on qubit 2, with ancilla qubit 0. Measure "
         "only the ancilla, into a 1-bit classical register.",
         "distribution", _circuit(3, _swap_test, clbits=1), {"target": {"0": 0.75, "1": 0.25}, "tol": 0.04}),
    Task("ms13_half_adder", "algorithm",
         "Build a quantum half adder: prepare inputs a = 1 on qubit 0 and b = 1 on qubit 1, compute the "
         "sum a XOR b into qubit 2 and the carry a AND b into qubit 3, then measure qubit 2 into classical "
         "bit 0 and qubit 3 into classical bit 1.",
         "distribution", _circuit(4, _half_adder, clbits=2), {"target": {"10": 1.0}, "tol": 0.02}),
    Task("ms14_bell_measurement", "algorithm",
         "Prepare the Bell state (|01> + |10>)/sqrt(2) on qubits 0 and 1, then measure it in the Bell basis "
         "by applying a CNOT (control 0, target 1) and an H on qubit 0, and measure qubit i into classical "
         "bit i.",
         "distribution", _circuit(2, _bell_measurement, clbits=2), {"target": {"10": 1.0}, "tol": 0.02}),

    # ---- applications ----
    Task("ap01_qaoa_triangle", "application",
         "Build the depth-1 QAOA state for MaxCut on the triangle graph with edges (0,1), (1,2) and (0,2), "
         "using gamma = 0.4 and beta = 0.7: start from |+>|+>|+>, apply the cost unitary "
         "exp(-i * gamma * sum over edges of Z_a Z_b), then the mixer exp(-i * beta * (X0 + X1 + X2)). "
         "Do not measure.",
         "statevector", _circuit(3, _qaoa_triangle)),
    Task("ap02_amplitude_encoding", "application",
         "Amplitude-encode the vector [1, 2, 3, 4] into 2 qubits: the amplitude of the basis state with "
         "integer value k (Qiskit order, k = q0 + 2*q1) must be proportional to entry k of the vector.",
         "statevector", lambda: _state([1, 2, 3, 4])),
    Task("ap03_angle_encoding", "application",
         "Angle-encode the data x = [0.1, 0.5, 0.9] into 3 qubits by applying RY(x_i) to qubit i.",
         "statevector", _circuit(3, _angle_encoding)),
    Task("ap04_bitflip_recovery", "application",
         "Demonstrate the 3-qubit bit-flip code without measurement: prepare RY(0.8)|0> on qubit 0, "
         "encode it into qubits 0-2, apply an X error to qubit 1, then decode and correct with CNOTs and a "
         "Toffoli so that qubit 0 ends in RY(0.8)|0> again.",
         "reduced_state", _ry_target(0.8), {"qubits": [0]}),
    Task("ap05_phaseflip_encode", "application",
         "Prepare RY(1.1)|0> = a|0> + b|1> on qubit 0 and encode it into the 3-qubit phase-flip code "
         "a|+++> + b|---> on qubits 0-2.",
         "statevector", _circuit(3, _phaseflip_encode)),
]

TASKS_BY_ID = {t.id: t for t in TASKS}
CATEGORIES = sorted({t.category for t in TASKS})


def select_tasks(ids: list[str] | None = None, category: str | None = None,
                 limit: int | None = None) -> list[Task]:
    """Pick tasks by id list, category, and/or a limit (applied last, in list order)."""
    tasks = TASKS
    if ids:
        unknown = [i for i in ids if i not in TASKS_BY_ID]
        if unknown:
            raise KeyError(f"unknown task ids: {', '.join(unknown)}")
        tasks = [TASKS_BY_ID[i] for i in ids]
    if category:
        tasks = [t for t in tasks if t.category == category]
    if limit is not None:
        tasks = tasks[:limit]
    return list(tasks)
