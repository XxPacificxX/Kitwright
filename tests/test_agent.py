import pytest

from ketwright.agent import (
    DONE_BUT_FAILED_PROMPT,
    NO_CODE_PROMPT,
    SYSTEM_PROMPT,
    extract_code,
    says_done,
    solve,
)
from ketwright.backends import BackendError, Reply, Usage
from tests.helpers import DONE, GHZ3, GHZ3_OLD_API, ScriptedBackend, fence, inprocess_run


def run_agent(*replies, max_rounds=4):
    backend = ScriptedBackend(list(replies))
    result = solve("Prepare a 3-qubit GHZ state.", backend, max_rounds=max_rounds, run=inprocess_run)
    return result, backend.conversations[0]


# --------------------------------------------------------------------------- code extraction

def test_extract_code_variants():
    assert extract_code("```python\nqc = 1\n```") == "qc = 1"
    assert extract_code("```\nqc = 2\n```") == "qc = 2"
    assert extract_code("```Python3\nqc = 3\n```") == "qc = 3"
    assert extract_code("no code here") is None


def test_extract_code_prefers_the_last_block_that_defines_qc():
    text = "```python\nqc = 'first'\n```\nand a helper:\n```python\ndef f():\n    pass\n```"
    assert extract_code(text) == "qc = 'first'"
    text2 = "```python\nqc = 'a'\n```\n```python\nqc = 'b'\n```"
    assert extract_code(text2) == "qc = 'b'"


def test_extract_code_handles_a_reply_cut_off_mid_block():
    assert extract_code("Sure:\n```python\nqc = QuantumCircuit(2)\nqc.h(") == "qc = QuantumCircuit(2)\nqc.h("


def test_says_done():
    assert says_done("VERDICT: DONE")
    assert says_done("Looks right.\nverdict:  done")
    assert not says_done("Not done yet")


# --------------------------------------------------------------------------- the loop

def test_system_prompt_states_the_contract():
    assert "`qc`" in SYSTEM_PROMPT and "Qubit 0 is the least significant bit" in SYSTEM_PROMPT


def test_correct_first_try_then_self_verifies():
    result, conv = run_agent(fence(GHZ3), DONE)
    assert result.status == "verified"
    assert len(result.attempts) == 1
    assert result.one_shot is result.final
    assert "Final statevector" in conv.received[1]  # the review prompt carries the execution report
    assert "|111>" in conv.received[1]
    assert result.usage.calls == 2 and result.first_call_usage.calls == 1


def test_fixes_its_own_runtime_error():
    result, conv = run_agent(fence(GHZ3_OLD_API), fence(GHZ3), DONE)
    assert result.status == "verified"
    assert [a.run.ok for a in result.attempts] == [False, True]
    assert result.one_shot is not None and not result.one_shot.run.ok
    assert "Your script failed" in conv.received[1] and "ImportError" in conv.received[1]


def test_max_rounds_one_is_plain_one_shot():
    result, conv = run_agent(fence(GHZ3), max_rounds=1)
    assert result.status == "out_of_rounds"
    assert len(conv.received) == 1


def test_done_after_a_failed_run_is_not_accepted():
    result, conv = run_agent(fence(GHZ3_OLD_API), DONE, fence(GHZ3), DONE)
    assert conv.received[2] == DONE_BUT_FAILED_PROMPT
    assert result.status == "verified" and result.final.run.ok


def test_reply_without_code_gets_a_nudge_and_no_one_shot_credit():
    result, conv = run_agent("I would use an H and two CNOTs.", fence(GHZ3), DONE)
    assert conv.received[1] == NO_CODE_PROMPT
    assert result.one_shot is None
    assert result.final.run.ok


def test_repeating_the_same_code_with_done_counts_as_verified():
    result, _ = run_agent(fence(GHZ3), fence(GHZ3) + "\nVERDICT: DONE")
    assert result.status == "verified" and len(result.attempts) == 1


def test_stops_after_max_rounds_of_code():
    result, conv = run_agent(*[fence(GHZ3_OLD_API)] * 3, max_rounds=3)
    assert result.status == "out_of_rounds"
    assert len(result.attempts) == 3 and len(conv.received) == 3


def test_refusal_stops_the_loop():
    result, _ = run_agent(Reply("", Usage(calls=1), "refusal"))
    assert result.status == "refused" and result.attempts == []


def test_non_fatal_backend_error_is_recorded_and_fatal_one_raises():
    result, _ = run_agent(fence(GHZ3), BackendError("rate limited"))
    assert result.status == "error" and "rate limited" in result.error
    assert len(result.attempts) == 1
    with pytest.raises(BackendError):
        run_agent(BackendError("bad key", fatal=True))
