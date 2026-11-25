"""The child-process runner (real subprocesses, no model calls)."""

from ketwright.sandbox import run_code
from tests.helpers import GHZ3, GHZ3_OLD_API


def test_good_script_returns_its_circuit():
    result = run_code(GHZ3)
    assert result.ok and result.error is None
    assert result.circuit.num_qubits == 3
    assert dict(result.circuit.count_ops()) == {"h": 1, "cx": 2}


def test_import_of_removed_api_is_reported():
    result = run_code(GHZ3_OLD_API)
    assert not result.ok
    assert "ImportError" in result.error and "execute" in result.error
    assert 'File "candidate.py"' in result.error


def test_missing_or_wrong_qc_is_reported():
    assert "did not define a top-level variable named `qc`" in run_code("x = 1").error
    assert "must be a qiskit.QuantumCircuit" in run_code("qc = 42").error


def test_prints_are_captured_and_feed_forward_circuits_round_trip():
    code = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.measure(0, 0)
with qc.if_test((qc.clbits[0], 1)):
    qc.x(1)
print("built", qc.num_qubits)
"""
    result = run_code(code)
    assert result.ok, result.error
    assert "built 2" in result.stdout


def test_timeout():
    result = run_code("while True:\n    pass\n", timeout=5)
    assert not result.ok and "did not finish" in result.error


def test_secrets_are_not_passed_to_generated_code(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    result = run_code("import os\nprint(os.environ.get('ANTHROPIC_API_KEY'))\nfrom qiskit import QuantumCircuit\nqc = QuantumCircuit(1)\n")
    assert result.ok
    assert result.stdout.strip() == "None"
