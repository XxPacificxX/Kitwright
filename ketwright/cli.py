"""Command line: `python -m ketwright <command>`.

  run "prepare a 5-qubit GHZ state"   solve one task and print each attempt
  eval [--limit N | --ids a,b]        run the evaluation set
  report results/<run>                 re-print a run's summary
  tasks                                list the evaluation tasks
  check-refs                           grade every reference solution (no model calls)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ketwright.backends import BackendError, make_backend


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama",
                   help="ollama (local, default) or anthropic (Claude API, billed)")
    p.add_argument("--model", help="model name (default: qwen2.5-coder:7b for ollama, claude-opus-5-5 for anthropic)")
    p.add_argument("--max-rounds", type=int, default=4, help="maximum code attempts per task (1 = one-shot)")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                   help="Claude effort level (anthropic backend only)")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama server URL")
    p.add_argument("--num-ctx", type=int, default=16384, help="Ollama context window in tokens")


def _backend(args: argparse.Namespace):
    return make_backend(args.backend, args.model, effort=args.effort, host=args.host, num_ctx=args.num_ctx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ketwright", description="Plain-English quantum tasks -> checked Qiskit code.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="solve one task")
    p_run.add_argument("task", nargs="+", help="the task in plain English")
    p_run.add_argument("--save", type=Path, help="write the final script to this file")
    p_run.add_argument("--quiet", action="store_true", help="print only the final script")
    _add_backend_args(p_run)

    p_eval = sub.add_parser("eval", help="run the evaluation set")
    p_eval.add_argument("--ids", help="comma-separated task ids")
    p_eval.add_argument("--category", help="only tasks in this category")
    p_eval.add_argument("--limit", type=int, help="only the first N selected tasks")
    p_eval.add_argument("--out", type=Path, help="output directory (default: results/<backend>-<model>)")
    p_eval.add_argument("--resume", action="store_true", help="continue an interrupted run in --out")
    p_eval.add_argument("--yes", action="store_true", help="skip the cost confirmation for paid backends")
    _add_backend_args(p_eval)

    p_report = sub.add_parser("report", help="print the summary of a finished run")
    p_report.add_argument("path", type=Path, help="a run directory or its results.jsonl")

    sub.add_parser("tasks", help="list the evaluation tasks")
    sub.add_parser("check-refs", help="grade every task's reference solution (no model calls)")
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        return {
            "run": _cmd_run,
            "eval": _cmd_eval,
            "report": _cmd_report,
            "tasks": _cmd_tasks,
            "check-refs": _cmd_check_refs,
        }[args.command](args)
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _cmd_run(args: argparse.Namespace) -> int:
    from ketwright.agent import solve

    task = " ".join(args.task)
    backend = _backend(args)
    if not args.quiet:
        print(f"Task: {task}\nBackend: {backend.name} / {backend.model}, up to {args.max_rounds} attempts\n")

    def show(kind: str, payload: dict) -> None:
        if args.quiet:
            return
        if kind == "attempt":
            a = payload["attempt"]
            print(f"--- attempt {a.number} ---\n{a.code}\n--- execution report ---\n{a.report}\n")
        elif kind == "reply" and "```" not in payload["text"]:
            print(f"--- model ---\n{payload['text'].strip()}\n")

    result = solve(task, backend, max_rounds=args.max_rounds, on_event=show)
    final = result.final
    outcome = {
        "verified": f"the agent checked its own result and declared it done after {len(result.attempts)} attempt(s)",
        "out_of_rounds": f"stopped after {len(result.attempts)} attempts without a DONE verdict",
        "out_of_calls": "stopped: the model kept replying without usable code",
        "refused": "the model declined the request",
        "error": f"stopped on a backend error: {result.error}",
    }[result.status]
    u = result.usage
    if args.quiet:
        if final:
            print(final.code)
    else:
        print(f"Result: {outcome}.")
        cost = "" if backend.name == "ollama" else f", cost ${u.cost_usd:.4f}"
        print(f"Usage: {u.calls} model calls, {u.input_tokens:,} input / {u.output_tokens:,} output tokens, "
              f"{u.seconds:.0f} s model time{cost}.")
    if final and args.save:
        args.save.write_text(final.code + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"Saved the final script to {args.save}")
    return 0 if final and final.run.ok else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    from ketwright.evaluate import estimate_anthropic_cost, render_markdown, run_eval, summarize
    from ketwright.tasks import select_tasks

    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    tasks = select_tasks(ids=ids, category=args.category, limit=args.limit)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 2

    if args.backend == "anthropic" and not args.yes:
        est = estimate_anthropic_cost(len(tasks), args.max_rounds)
        print(f"This runs {len(tasks)} tasks on the Claude API and is billed to your account.\n"
              f"Rough estimate: ~${est['typical_usd']:.2f} typical, up to ~${est['worst_usd']:.2f} if every task "
              f"uses all {args.max_rounds} attempts (assumptions in ketwright/evaluate.py).")
        if not sys.stdin.isatty() or input("Continue? [y/N] ").strip().lower() != "y":
            print("Cancelled. Pass --yes to skip this prompt.")
            return 1

    backend = _backend(args)
    out = args.out or Path("results") / re.sub(r"[^A-Za-z0-9._-]+", "-", f"{backend.name}-{backend.model}")
    print(f"Evaluating {len(tasks)} tasks with {backend.name} / {backend.model}, up to {args.max_rounds} "
          f"attempts each -> {out}")
    records = run_eval(tasks, backend, args.max_rounds, out, resume=args.resume)
    print()
    print(render_markdown(summarize(records)))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from ketwright.evaluate import render_markdown, summarize

    path = args.path / "results.jsonl" if args.path.is_dir() else args.path
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(render_markdown(summarize(records)))
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    from ketwright.tasks import TASKS

    for t in TASKS:
        print(f"{t.id:32} {t.category:12} {t.kind:14} {t.prompt}")
    print(f"\n{len(TASKS)} tasks")
    return 0


def _cmd_check_refs(args: argparse.Namespace) -> int:
    from ketwright.checks import run_check
    from ketwright.tasks import TASKS

    failures = 0
    for t in TASKS:
        r = run_check(t, t.example())
        failures += not r.passed
        print(f"{'ok  ' if r.passed else 'FAIL'} {t.id:32} {r.detail}")
    print(f"\n{len(TASKS) - failures}/{len(TASKS)} reference solutions pass their checks")
    return 1 if failures else 0
