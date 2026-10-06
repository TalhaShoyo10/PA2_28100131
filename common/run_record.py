"""Reproducible run records and cross-account experiment status.

Every experiment writes to results/<task_dir>/<experiment_id>/:
    status.json        running | done | failed, worker id, heartbeat (shared via Google Drive)
    run_manifest.json  git commit, command, seed, runtime, timestamps, wall clock, peak VRAM
    config.json        fully merged configuration actually used
    command.txt        exact command line
    metrics.json       final metrics (written by finish())

Status rules (EXECUTION.md: never overwrite a finished or failed run):
    done                       -> refuse to rerun (RunAlreadyDone)
    running, fresh heartbeat   -> refuse (another Colab account is working on it)
    running stale / failed     -> archive the old directory as <id>__attempt<N>, then start fresh
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from common.data import repo_path
from common.logging_utils import save_json

STALE_AFTER_SECONDS = 30 * 60


class RunAlreadyDone(RuntimeError):
    pass


class RunInProgress(RuntimeError):
    pass


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_path("."), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    # results/ and outputs/ are Drive symlinks on Colab, so only tracked files count as "dirty".
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked_files": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def runtime_info() -> dict:
    import peft
    import transformers

    try:
        import trl
        trl_version = trl.__version__
    except Exception:
        trl_version = None
    info = {
        "worker": os.environ.get("PA2_WORKER", socket.gethostname()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl_version,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gib"] = round(props.total_memory / 2**30, 2)
    return info


def _command_line() -> str:
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    argv = ["python", "-m", spec.name, *sys.argv[1:]] if spec else ["python", *sys.argv]
    return " ".join(shlex.quote(a) for a in argv)


def experiment_id(task: str, condition: str, seed: int) -> str:
    return f"{task}_{condition}_seed{seed}"


class RunRecord:
    def __init__(self, results_dir: str | Path, exp_id: str, config: dict, extra: dict | None = None):
        self.exp_id = exp_id
        self.dir = repo_path(results_dir) / exp_id
        self.config = config
        self.extra = extra or {}
        self.manifest: dict = {}
        self._t0 = None

    # ---- status -------------------------------------------------------------------------
    @property
    def status_path(self) -> Path:
        return self.dir / "status.json"

    def read_status(self) -> dict | None:
        if not self.status_path.exists():
            return None
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def _write_status(self, state: str, **fields) -> None:
        save_json(self.status_path, {
            "experiment_id": self.exp_id,
            "state": state,
            "worker": os.environ.get("PA2_WORKER", socket.gethostname()),
            "updated_unix": time.time(),
            "updated_at": _now_iso(),
            **fields,
        })

    def is_done(self) -> bool:
        st = self.read_status()
        return bool(st and st["state"] == "done")

    def _archive_previous_attempt(self) -> Path:
        n = 1
        while (self.dir.parent / f"{self.exp_id}__attempt{n}").exists():
            n += 1
        target = self.dir.parent / f"{self.exp_id}__attempt{n}"
        self.dir.rename(target)
        return target

    # ---- lifecycle ----------------------------------------------------------------------
    def start(self) -> "RunRecord":
        st = self.read_status()
        if st is not None:
            if st["state"] == "done":
                raise RunAlreadyDone(f"{self.exp_id} is already done ({self.dir}).")
            age = time.time() - float(st.get("updated_unix", 0))
            if st["state"] == "running" and age < STALE_AFTER_SECONDS:
                raise RunInProgress(
                    f"{self.exp_id} is running on worker {st.get('worker')} "
                    f"(heartbeat {age/60:.1f} min ago). Not starting a second copy."
                )
            archived = self._archive_previous_attempt()
            print(f"[run_record] previous {st['state']} attempt archived to {archived}")
        elif self.dir.exists() and any(self.dir.iterdir()):
            archived = self._archive_previous_attempt()
            print(f"[run_record] unrecorded previous directory archived to {archived}")

        self.dir.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        command = _command_line()
        self.manifest = {
            "experiment_id": self.exp_id,
            "started_at": _now_iso(),
            "git": git_state(),
            "command": command,
            "seed": self.config.get("seed"),
            "runtime": runtime_info(),
            **self.extra,
        }
        (self.dir / "command.txt").write_text(command + "\n", encoding="utf-8")
        save_json(self.dir / "config.json", self.config)
        save_json(self.dir / "run_manifest.json", self.manifest)
        self._write_status("running", step=0)
        return self

    def heartbeat(self, **progress) -> None:
        self._write_status("running", **progress)

    def peak_vram_gib(self) -> float | None:
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 2**30, 3)

    def finish(self, metrics: dict, artifacts: dict | None = None) -> None:
        wall = time.perf_counter() - self._t0
        self.manifest.update({
            "ended_at": _now_iso(),
            "wall_clock_seconds": round(wall, 1),
            "peak_vram_allocated_gib": self.peak_vram_gib(),
            "artifacts": artifacts or {},
        })
        save_json(self.dir / "metrics.json", metrics)
        save_json(self.dir / "run_manifest.json", self.manifest)
        self._write_status("done", wall_clock_seconds=round(wall, 1))

    def fail(self, exc: BaseException) -> None:
        self.manifest.update({
            "ended_at": _now_iso(),
            "wall_clock_seconds": round(time.perf_counter() - self._t0, 1) if self._t0 else None,
            "peak_vram_allocated_gib": self.peak_vram_gib(),
        })
        save_json(self.dir / "run_manifest.json", self.manifest)
        (self.dir / "error.txt").write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
        self._write_status("failed", error=f"{type(exc).__name__}: {exc}"[:500])

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.fail(exc)
        return False
