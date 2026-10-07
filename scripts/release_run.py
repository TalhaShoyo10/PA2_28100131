"""Mark a stuck "running" experiment as failed so the next start archives it and restarts.

Use only when the session that ran it is gone (Colab disconnect / runtime expired). The run's
directory is kept; the next start moves it to <id>__attemptN as evidence.

    python -m scripts.release_run task2_ppo_eval_fork_eps005_kl010_seed6304
"""
from __future__ import annotations

import argparse
import json
import os
import time

from common.data import repo_path
from common.logging_utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_ids", nargs="+")
    args = ap.parse_args()
    root = repo_path(os.environ.get("PA2_RESULTS_ROOT", "results"))
    for exp_id in args.experiment_ids:
        hits = [p for p in root.glob(f"*/{exp_id}/status.json")]
        if len(hits) != 1:
            print(f"{exp_id}: found {len(hits)} status files under {root}; nothing changed")
            continue
        st = json.loads(hits[0].read_text(encoding="utf-8"))
        if st["state"] != "running":
            print(f"{exp_id}: state is {st['state']}, not running; nothing changed")
            continue
        age = (time.time() - float(st.get("updated_unix", 0))) / 60
        st.update({
            "state": "failed",
            "error": f"released manually: session lost (was running on {st.get('worker')}, heartbeat {age:.1f} min old)",
            "updated_unix": time.time(),
        })
        save_json(hits[0], st)
        print(f"{exp_id}: released (it will be archived and restarted on the next run)")


if __name__ == "__main__":
    main()
