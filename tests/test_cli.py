import ketwright.cli as cli
from ketwright.cli import main
from tests.helpers import DONE, GHZ3, ScriptedBackend, fence


def test_tasks_lists_all_fifty(capsys):
    assert main(["tasks"]) == 0
    out = capsys.readouterr().out
    assert "sp04_ghz5" in out and "50 tasks" in out


def test_paid_eval_asks_first_and_cancels_without_a_terminal(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_backend", lambda args: (_ for _ in ()).throw(AssertionError("must not build")))
    assert main(["eval", "--backend", "anthropic", "--limit", "2"]) == 1
    out = capsys.readouterr().out
    assert "billed to your account" in out and "Cancelled" in out


def test_run_prints_attempts_and_saves_the_script(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_backend", lambda args: ScriptedBackend([fence(GHZ3), DONE]))
    target = tmp_path / "ghz.py"
    assert main(["run", "prepare", "a", "3-qubit", "GHZ", "state", "--save", str(target)]) == 0
    out = capsys.readouterr().out
    assert "--- attempt 1 ---" in out and "declared it done" in out
    assert "qc.cx(1, 2)" in target.read_text(encoding="utf-8")


def test_eval_and_report_end_to_end_with_a_scripted_model(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_backend", lambda args: ScriptedBackend([fence(GHZ3), DONE]))
    assert main(["eval", "--ids", "sp03_ghz3", "--out", str(tmp_path), "--max-rounds", "2"]) == 0
    assert "| Pass rate | 1/1" in capsys.readouterr().out
    assert main(["report", str(tmp_path)]) == 0
    assert "| Pass rate | 1/1" in capsys.readouterr().out
