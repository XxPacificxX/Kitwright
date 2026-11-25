import json

import pytest

from ketwright.evaluate import (
    estimate_anthropic_cost,
    mcnemar_exact,
    render_markdown,
    run_eval,
    summarize,
    wilson_interval,
)
from ketwright.tasks import select_tasks
from tests.helpers import BASIS_1010_RIGHT, BASIS_1010_WRONG_ORDER, DONE, GHZ3, ScriptedBackend, fence


def two_task_run(tmp_path, resume=False, backend=None):
    tasks = select_tasks(ids=["sp03_ghz3", "sp07_basis_1010"])
    backend = backend or ScriptedBackend(
        [fence(GHZ3), DONE],  # right first time
        # wrong bit order first, then fixed after reading the execution report
        [fence(BASIS_1010_WRONG_ORDER), fence(BASIS_1010_RIGHT), DONE],
    )
    from tests.helpers import inprocess_run

    records = run_eval(tasks, backend, max_rounds=3, out_dir=tmp_path, resume=resume, run=inprocess_run,
                       log=lambda _: None)
    return records, backend


def test_eval_grades_one_shot_and_self_correcting_separately(tmp_path):
    records, _ = two_task_run(tmp_path)
    ghz, basis = records
    assert ghz["one_shot_pass"] and ghz["final_pass"] and ghz["status"] == "verified"
    assert not basis["one_shot_pass"] and basis["final_pass"]
    assert basis["pass_by_round"] == [False, True, True]
    assert "fidelity" in basis["attempts"][0]["detail"]
    assert basis["usage"]["calls"] == 3 and basis["one_shot_usage"]["calls"] == 1

    s = summarize(records)
    assert s["one_shot"]["passed"] == 1 and s["self_correcting"]["passed"] == 2
    assert s["fixed_by_self_correction"] == 1 and s["broken_by_self_correction"] == 0
    assert s["pass_rate_by_round_cap"] == [0.5, 1.0, 1.0]
    assert s["self_check"] == {"said_done": 2, "said_done_and_correct": 2, "said_done_but_wrong": 0}


def test_eval_writes_jsonl_and_summaries(tmp_path):
    two_task_run(tmp_path)
    lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["task_id"] for line in lines] == ["sp03_ghz3", "sp07_basis_1010"]
    assert json.loads((tmp_path / "summary.json").read_text())["n_tasks"] == 2
    assert "| Pass rate |" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "config.json").read_text())["max_rounds"] == 3


def test_resume_skips_finished_tasks_and_refuses_to_overwrite(tmp_path):
    two_task_run(tmp_path)
    with pytest.raises(FileExistsError):
        two_task_run(tmp_path)
    records, backend = two_task_run(tmp_path, resume=True, backend=ScriptedBackend())  # no scripts: no calls
    assert len(records) == 2 and backend.conversations == []


def test_statistics():
    lo, hi = wilson_interval(25, 50)
    assert lo == pytest.approx(0.366, abs=1e-3) and hi == pytest.approx(0.634, abs=1e-3)
    assert mcnemar_exact(9, 1) == pytest.approx(2 * 11 / 1024)
    assert mcnemar_exact(0, 0) == 1.0


def test_markdown_for_local_runs_reports_zero_cost(tmp_path):
    records, _ = two_task_run(tmp_path)
    for r in records:
        r["backend"] = "ollama"
    md = render_markdown(summarize(records))
    assert "$0 (local)" in md and "fixed 1 tasks and broke 0" in md


def test_cost_estimate_scales_with_tasks():
    small, big = estimate_anthropic_cost(5, 4), estimate_anthropic_cost(50, 4)
    assert big["typical_usd"] == pytest.approx(10 * small["typical_usd"])
    assert big["worst_usd"] > big["typical_usd"] > 0
