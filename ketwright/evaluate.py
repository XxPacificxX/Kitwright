"""Evaluation harness: one-shot vs self-correcting pass rates, tokens, time and cost.

One run of the agent gives both numbers. The agent's first reply is exactly
what a one-shot agent would return (same prompt, deterministic decoding), so
"one-shot" scores that first reply and "self-correcting" scores the code the
agent ends with after its own review rounds. Both are graded by the same
hidden checks, which the agent never sees.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ketwright.agent import solve
from ketwright.backends import Backend
from ketwright.checks import CheckResult, run_check
from ketwright.sandbox import RunResult, run_code
from ketwright.tasks import Task

Logger = Callable[[str], None]


def evaluate_task(task: Task, backend: Backend, max_rounds: int,
                  run: Callable[[str], RunResult] = run_code) -> dict[str, Any]:
    """Run the agent on one task and grade every attempt with the hidden check."""
    start = time.perf_counter()
    result = solve(task.prompt, backend, max_rounds=max_rounds, run=run)

    graded = []
    for attempt in result.attempts:
        if attempt.run.ok:
            check = run_check(task, attempt.run.circuit)
        else:
            first_line = (attempt.run.error or "script failed").strip().splitlines()[-1][:200]
            check = CheckResult(False, f"script failed: {first_line}")
        graded.append({
            "n": attempt.number,
            "call": attempt.call,
            "ran": attempt.run.ok,
            "pass": check.passed,
            "detail": check.detail,
            "code": attempt.code,
        })

    one_shot_pass = bool(result.one_shot is not None and graded[0]["pass"])
    final_pass = bool(graded and graded[-1]["pass"])
    by_round = [one_shot_pass]
    for k in range(2, max_rounds + 1):
        by_round.append(bool(graded and graded[min(k, len(graded)) - 1]["pass"]))

    return {
        "task_id": task.id,
        "category": task.category,
        "backend": backend.name,
        "model": backend.model,
        "max_rounds": max_rounds,
        "one_shot_pass": one_shot_pass,
        "one_shot_detail": graded[0]["detail"] if result.one_shot is not None else "no code in the first reply",
        "final_pass": final_pass,
        "final_detail": graded[-1]["detail"] if graded else "no code was produced",
        "pass_by_round": by_round,
        "status": result.status,
        "n_attempts": len(graded),
        "usage": result.usage.to_dict(),
        "one_shot_usage": result.first_call_usage.to_dict(),
        "seconds": round(time.perf_counter() - start, 2),
        "error": result.error,
        "attempts": graded,
        "transcript": result.transcript,
    }


def run_eval(tasks: list[Task], backend: Backend, max_rounds: int, out_dir: Path, resume: bool = False,
             run: Callable[[str], RunResult] = run_code, log: Logger = print) -> list[dict[str, Any]]:
    """Evaluate `tasks`, appending one JSON line per task to out_dir/results.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        if not resume:
            raise FileExistsError(f"{results_path} already exists; pass --resume to continue it or pick another --out")
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["task_id"]] = record

    config = {
        "backend": backend.name,
        "model": backend.model,
        "max_rounds": max_rounds,
        "tasks": [t.id for t in tasks],
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with results_path.open("a", encoding="utf-8") as fh:
        for i, task in enumerate(tasks, 1):
            if task.id in done:
                log(f"[{i}/{len(tasks)}] {task.id}: already done, skipping")
                continue
            record = evaluate_task(task, backend, max_rounds, run=run)
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            done[task.id] = record
            log(f"[{i}/{len(tasks)}] {task.id}: one-shot {_mark(record['one_shot_pass'])} -> "
                f"final {_mark(record['final_pass'])} ({record['n_attempts']} attempts, {record['status']}, "
                f"{record['seconds']:.0f}s)")

    records = [done[t.id] for t in tasks if t.id in done]
    summary = summarize(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return records


def _mark(passed: bool) -> str:
    return "PASS" if passed else "fail"


# --------------------------------------------------------------------------- statistics

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail changes."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class _Rate:
    passed: int
    n: int

    def as_dict(self) -> dict[str, Any]:
        lo, hi = wilson_interval(self.passed, self.n)
        return {"passed": self.passed, "n": self.n, "rate": self.passed / self.n if self.n else 0.0,
                "ci95": [round(lo, 3), round(hi, 3)]}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n_tasks": 0}
    one = sum(r["one_shot_pass"] for r in records)
    final = sum(r["final_pass"] for r in records)
    fixed = sum(1 for r in records if r["final_pass"] and not r["one_shot_pass"])
    broken = sum(1 for r in records if r["one_shot_pass"] and not r["final_pass"])
    rounds = max(r["max_rounds"] for r in records)
    by_round = [
        _mean([float(r["pass_by_round"][min(k, len(r["pass_by_round"]) - 1)]) for r in records])
        for k in range(rounds)
    ]
    categories: dict[str, dict[str, Any]] = {}
    for cat in sorted({r["category"] for r in records}):
        rs = [r for r in records if r["category"] == cat]
        categories[cat] = {
            "n": len(rs),
            "one_shot": sum(r["one_shot_pass"] for r in rs),
            "self_correcting": sum(r["final_pass"] for r in rs),
        }
    verified = [r for r in records if r["status"] == "verified"]

    def usage_means(key: str) -> dict[str, float]:
        return {
            "input_tokens": round(_mean([r[key]["input_tokens"] for r in records]), 1),
            "output_tokens": round(_mean([r[key]["output_tokens"] for r in records]), 1),
            "cache_read_tokens": round(_mean([r[key]["cache_read_tokens"] for r in records]), 1),
            "model_calls": round(_mean([r[key]["calls"] for r in records]), 2),
            "model_seconds": round(_mean([r[key]["seconds"] for r in records]), 1),
            "cost_usd": round(_mean([r[key]["cost_usd"] for r in records]), 5),
        }

    return {
        "backend": records[0]["backend"],
        "model": records[0]["model"],
        "max_rounds": rounds,
        "n_tasks": n,
        "one_shot": _Rate(one, n).as_dict(),
        "self_correcting": _Rate(final, n).as_dict(),
        "fixed_by_self_correction": fixed,
        "broken_by_self_correction": broken,
        "mcnemar_p": round(mcnemar_exact(fixed, broken), 4),
        "pass_rate_by_round_cap": [round(x, 3) for x in by_round],
        "by_category": categories,
        "per_task_one_shot": usage_means("one_shot_usage"),
        "per_task_self_correcting": usage_means("usage"),
        "mean_attempts": round(_mean([r["n_attempts"] for r in records]), 2),
        "mean_wall_seconds": round(_mean([r["seconds"] for r in records]), 1),
        "self_check": {
            "said_done": len(verified),
            "said_done_and_correct": sum(r["final_pass"] for r in verified),
            "said_done_but_wrong": sum(not r["final_pass"] for r in verified),
        },
        "statuses": {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
        "errors": sum(1 for r in records if r.get("error")),
    }


def render_markdown(s: dict[str, Any]) -> str:
    if not s.get("n_tasks"):
        return "No results.\n"
    one, sc = s["one_shot"], s["self_correcting"]
    u1, us = s["per_task_one_shot"], s["per_task_self_correcting"]

    def rate(r: dict[str, Any]) -> str:
        lo, hi = r["ci95"]
        return f"{r['passed']}/{r['n']} ({r['rate']:.0%}, 95% CI {lo:.0%}-{hi:.0%})"

    def money(x: float) -> str:
        return "$0 (local)" if s["backend"] == "ollama" else f"${x:.4f}"

    lines = [
        f"## {s['backend']} / {s['model']}: {s['n_tasks']} tasks, up to {s['max_rounds']} attempts",
        "",
        "| | One-shot | Self-correcting |",
        "|---|---|---|",
        f"| Pass rate | {rate(one)} | {rate(sc)} |",
        f"| Model calls per task | {u1['model_calls']:.2f} | {us['model_calls']:.2f} |",
        f"| Input tokens per task | {u1['input_tokens']:,.0f} | {us['input_tokens']:,.0f} |",
        f"| Output tokens per task | {u1['output_tokens']:,.0f} | {us['output_tokens']:,.0f} |",
        f"| Model time per task | {u1['model_seconds']:.1f} s | {us['model_seconds']:.1f} s |",
        f"| Cost per task | {money(u1['cost_usd'])} | {money(us['cost_usd'])} |",
        "",
        f"Self-correction fixed {s['fixed_by_self_correction']} tasks and broke "
        f"{s['broken_by_self_correction']} (exact McNemar p = {s['mcnemar_p']}).",
        "",
        "Pass rate when the agent is capped at k attempts: "
        + ", ".join(f"k={k + 1}: {x:.0%}" for k, x in enumerate(s["pass_rate_by_round_cap"])) + ".",
        "",
        f"Self-check: the agent declared `VERDICT: DONE` on {s['self_check']['said_done']} tasks; "
        f"{s['self_check']['said_done_but_wrong']} of those were actually wrong.",
        "",
        "| Category | Tasks | One-shot | Self-correcting |",
        "|---|---|---|---|",
    ]
    for cat, c in s["by_category"].items():
        lines.append(f"| {cat} | {c['n']} | {c['one_shot']} | {c['self_correcting']} |")
    lines += ["", f"Stop reasons: {', '.join(f'{k} {v}' for k, v in s['statuses'].items())}. "
                  f"Mean attempts {s['mean_attempts']}, mean wall time {s['mean_wall_seconds']:.0f} s per task.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- cost estimate

# Assumed Claude Opus 5.5 token use per task at effort "high" (thinking counts as output).
# These are planning assumptions, not measurements; replace them with a measured slice.
EST_CALLS_TYPICAL, EST_CALLS_WORST = 2.5, 8
EST_INPUT_PER_CALL, EST_OUTPUT_PER_CALL = 3000, 1500


def estimate_anthropic_cost(n_tasks: int, max_rounds: int, price_in: float = 4.0,
                            price_out: float = 20.0) -> dict[str, float]:
    per_call = (EST_INPUT_PER_CALL * price_in + EST_OUTPUT_PER_CALL * price_out) / 1e6
    worst_calls = min(EST_CALLS_WORST, 2 * max_rounds)
    return {
        "per_call_usd": per_call,
        "typical_usd": n_tasks * EST_CALLS_TYPICAL * per_call,
        "worst_usd": n_tasks * worst_calls * per_call * 1.5,  # later calls carry longer histories
    }
