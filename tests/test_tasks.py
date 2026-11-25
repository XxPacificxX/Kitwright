import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

from ketwright.checks import run_check
from ketwright.tasks import CATEGORIES, TASKS, TASKS_BY_ID, dft_matrix, select_tasks


def test_fifty_unique_tasks_in_four_categories():
    assert len(TASKS) == 50
    assert len(TASKS_BY_ID) == 50
    assert CATEGORIES == ["algorithm", "application", "state-prep", "unitary"]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_known_good_solution_passes_its_check(task):
    result = run_check(task, task.example())
    assert result.passed, result.detail


@pytest.mark.parametrize("task", [t for t in TASKS if t.solution], ids=lambda t: t.id)
def test_gate_restricted_tasks_reject_the_library_gate(task):
    # The reference uses the high-level gate the task forbids, so it must fail the restriction.
    assert not run_check(task, task.reference()).passed


def test_qftgate_matches_the_dft_written_in_the_prompts():
    # The QPE references rely on QFTGate following the same convention as the un01/un02 prompts.
    assert np.allclose(Operator(QFTGate(3)).data, dft_matrix(3))


def test_cyclic_shift_reference_moves_qubit0_to_qubit1():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.compose(TASKS_BY_ID["un10_cyclic_shift3"].reference(), inplace=True)
    assert Statevector(qc).probabilities_dict() == pytest.approx({"010": 1.0})


def test_select_tasks_by_id_category_and_limit():
    assert [t.id for t in select_tasks(ids=["sp03_ghz3", "un01_qft3"])] == ["sp03_ghz3", "un01_qft3"]
    assert {t.category for t in select_tasks(category="unitary")} == {"unitary"}
    assert len(select_tasks(limit=3)) == 3
    with pytest.raises(KeyError):
        select_tasks(ids=["nope"])
