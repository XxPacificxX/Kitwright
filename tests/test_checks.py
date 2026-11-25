"""The hidden checks must reject plausible wrong answers, not just accept right ones."""

from math import pi

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.synthesis import synth_qft_full

from ketwright.checks import run_check
from ketwright.tasks import TASKS_BY_ID, _grover


def check(task_id, qc):
    return run_check(TASKS_BY_ID[task_id], qc)


def ghz3():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


def test_missing_entangling_gate_fails():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    assert not check("sp03_ghz3", qc).passed


def test_wrong_qubit_count_fails():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    result = check("sp03_ghz3", qc)
    assert not result.passed and "expected 3 qubits" in result.detail


def test_final_measurements_are_ignored_by_state_checks():
    qc = ghz3()
    qc.measure_all()
    assert check("sp03_ghz3", qc).passed


def test_mid_circuit_measurement_fails_state_check():
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    assert not check("sp03_ghz3", qc).passed


def test_bit_order_matters():
    qc = QuantumCircuit(4)
    qc.x([0, 2])  # |0101>, the reversed reading of |1010>
    assert not check("sp07_basis_1010", qc).passed


def test_relative_phase_matters_but_global_phase_does_not():
    plus = QuantumCircuit(3)
    plus.h(0)
    plus.cx(0, 1)
    plus.cx(1, 2)
    assert not check("sp11_ghz_minus", plus).passed
    minus = plus.copy()
    minus.z(0)
    assert check("sp11_ghz_minus", minus).passed
    minus.global_phase = 1.234
    assert check("sp11_ghz_minus", minus).passed


def test_forbidden_gate_fails_and_custom_gates_are_inspected():
    direct = QuantumCircuit(2)
    direct.swap(0, 1)
    assert "outside the allowed set" in check("un03_swap_from_cx", direct).detail

    inner = QuantumCircuit(2, name="my_swap")
    inner.cx(0, 1)
    inner.cx(1, 0)
    inner.cx(0, 1)
    wrapped = QuantumCircuit(2)
    wrapped.append(inner.to_gate(), [0, 1])
    assert check("un03_swap_from_cx", wrapped).passed

    hidden = QuantumCircuit(2, name="sneaky")
    hidden.swap(0, 1)
    wrapped_bad = QuantumCircuit(2)
    wrapped_bad.append(hidden.to_gate(), [0, 1])
    assert not check("un03_swap_from_cx", wrapped_bad).passed


def test_qft_without_final_swaps_fails():
    assert check("un01_qft3", synth_qft_full(3, do_swaps=True)).passed
    assert not check("un01_qft3", synth_qft_full(3, do_swaps=False)).passed


def test_wrong_rotation_convention_fails():
    # exp(-i 0.3 ZZ) is RZZ(0.6); RZZ(0.3) is the classic factor-of-two slip.
    qc = QuantumCircuit(2)
    qc.rzz(0.3, 0, 1)
    assert not check("un15_zz_evolution", qc).passed


def test_too_few_grover_iterations_fails():
    assert not check("ms01_grover3_101", _grover(3, "101", 1)).passed  # P(101) is about 0.78


def test_extra_measured_bits_fail():
    qc = _grover(3, "101", 2)
    qc.add_register(ClassicalRegister(1, "extra"))
    qc.measure(0, 3)
    result = check("ms01_grover3_101", qc)
    assert not result.passed and "classical bits" in result.detail


def test_loose_classical_bits_are_read_in_index_order():
    qc = _grover(3, "101", 2)
    qc.add_bits([type(qc.clbits[0])()])  # a classical bit that belongs to no register
    qc.measure(0, 3)
    result = check("ms01_grover3_101", qc)
    assert not result.passed and "[4] classical bits" in result.detail


def test_teleportation_without_corrections_fails():
    qc = QuantumCircuit(3, 3)
    qc.ry(pi / 3, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    qc.measure(2, 2)
    assert not check("ms06_teleportation", qc).passed


def test_reduced_state_check_allows_extra_qubits_but_not_wrong_states():
    good = QuantumCircuit(4)
    good.ry(0.8, 0)
    good.h(3)  # an unrelated extra qubit
    assert check("ap04_bitflip_recovery", good).passed
    bad = QuantumCircuit(3)
    bad.ry(0.4, 0)
    assert not check("ap04_bitflip_recovery", bad).passed


def test_unbound_parameters_and_missing_circuit_fail():
    qc = QuantumCircuit(1)
    qc.ry(Parameter("theta"), 0)
    assert "unbound parameters" in check("sp09_bloch", qc).detail
    assert not check("sp09_bloch", None).passed
