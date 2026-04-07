from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import yaml

from graders import validate_baseline_reproducibility, validate_rl_assets, validate_task_graders
from tasks import AVAILABLE_TASKS
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
REQUIRED_ENV_VARS = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]


def _check_openenv_config() -> Dict[str, object]:
    config_path = ROOT / "openenv.yaml"
    if not config_path.exists():
        return {"ok": False, "reason": "openenv.yaml is missing"}

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required_paths = [
        ("environment", "name"),
        ("environment", "description"),
        ("api", "step_implementation"),
        ("api", "reset_implementation"),
        ("api", "state_implementation"),
        ("spaces", "action"),
        ("spaces", "observation"),
    ]
    missing = [
        ".".join(path)
        for path in required_paths
        if not isinstance(config, dict)
        or path[0] not in config
        or path[1] not in config.get(path[0], {})
    ]
    return {"ok": not missing, "missing": missing}


def _check_required_files() -> Dict[str, object]:
    required_files = ["Dockerfile", "inference.py", "openenv.yaml"]
    statuses = {name: (ROOT / name).exists() for name in required_files}
    return {"ok": all(statuses.values()), "files": statuses}


def _check_required_env_vars() -> Dict[str, object]:
    statuses = {name: bool(os.environ.get(name)) for name in REQUIRED_ENV_VARS}
    return {"ok": all(statuses.values()), "variables": statuses}


def main() -> int:
    report = {
        "required_files": _check_required_files(),
        "openenv_config": _check_openenv_config(),
        "required_env_vars": _check_required_env_vars(),
        "task_inventory": {
            "task_count": len(AVAILABLE_TASKS),
            "minimum_tasks_met": len(AVAILABLE_TASKS) >= 3,
            "tasks": AVAILABLE_TASKS,
        },
        "task_graders": validate_task_graders(),
        "baseline_reproducibility": validate_baseline_reproducibility(),
        "rl_assets": validate_rl_assets(),
    }

    report["pass"] = all(
        [
            report["required_files"]["ok"],
            report["openenv_config"]["ok"],
            report["task_inventory"]["minimum_tasks_met"],
            report["task_graders"]["minimum_tasks_met"],
            report["task_graders"]["all_scores_in_range"],
            report["baseline_reproducibility"]["reproducible"],
        ]
    )

    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
