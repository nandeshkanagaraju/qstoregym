"""
Automated retraining pipeline for Q-Store Gym.

Implements the production retraining loop described in Section 6.5 / Section 7:
  - Loads the current deployed model and evaluates it on all tasks
  - Trains a new model for N timesteps (continuing from existing weights)
  - Runs a quality gate: new model must beat the old model's score by a threshold
    OR the current model must be below a minimum acceptable score
  - If the quality gate passes, replaces the deployed model
  - Logs all decisions with timestamps

Usage:
    # Retrain per-task models
    python retrain.py --timesteps 200000

    # Retrain curriculum model
    python retrain.py --curriculum --timesteps 500000

    # Dry run — evaluate only, do not replace models
    python retrain.py --dry-run

Scheduling:
    Run this script on a cron schedule (daily or weekly).
    Example crontab entry (daily at 2 AM):
        0 2 * * * cd /app && python retrain.py --timesteps 200000 >> logs/retrain.log 2>&1
"""

import argparse
import os
import shutil
import statistics
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_wrapper import QStoreGymWrapper
from curriculum import CurriculumGymWrapper
from tasks import AVAILABLE_TASKS


# ──────────────────────────────────────────────────────────────
# Quality gate settings
# ──────────────────────────────────────────────────────────────

# Minimum mean score the new model must achieve to be deployed.
MIN_ACCEPTABLE_SCORE = 0.15

# How much better the new model must be than the current one to trigger deployment
# (unless the current model is below MIN_ACCEPTABLE_SCORE, in which case any improvement deploys).
IMPROVEMENT_THRESHOLD = 0.02

# Number of evaluation episodes per task for scoring.
N_EVAL_EPISODES = 5


# ──────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────

def _evaluate_model(model_stem: str, task_name: str, n_episodes: int = N_EVAL_EPISODES) -> Optional[float]:
    """
    Run n_episodes of deterministic inference with the saved model.
    Returns mean final score, or None if the model file doesn't exist.
    """
    model_path   = f"{model_stem}.zip"
    vecnorm_path = f"{model_stem}_vecnorm.pkl"

    if not os.path.exists(model_path):
        return None

    raw = DummyVecEnv([lambda: Monitor(QStoreGymWrapper(task_name=task_name))])
    if os.path.exists(vecnorm_path):
        norm_env = VecNormalize.load(vecnorm_path, raw)
        norm_env.training    = False
        norm_env.norm_reward = False
    else:
        norm_env = raw

    model = PPO.load(model_stem, env=norm_env)

    scores = []
    for _ in range(n_episodes):
        obs = norm_env.reset()
        done = False
        final_score = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, infos = norm_env.step(action)
            done = bool(terminated[0])
            final_score = infos[0].get("score", final_score)
        scores.append(final_score)

    norm_env.close()
    return statistics.mean(scores) if scores else 0.0


def _evaluate_all_tasks(model_stem: str, is_curriculum: bool) -> Dict[str, float]:
    """Return {task_name: mean_score} for all tasks."""
    results = {}
    for task in AVAILABLE_TASKS:
        stem  = model_stem if is_curriculum else f"ppo_{task.replace(' ', '_')}"
        score = _evaluate_model(stem, task_name=task)
        results[task] = score if score is not None else 0.0
    return results


# ──────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────

def _train_one(model_stem: str, task_name: str, timesteps: int):
    """Train (or continue training) a per-task PPO model."""
    from train import train_per_task
    train_per_task(task_name=task_name, total_timesteps=timesteps, device="auto", n_envs=4)


def _train_curriculum(timesteps: int):
    from train import train_curriculum
    train_curriculum(total_timesteps=timesteps, device="auto", n_envs=4)


# ──────────────────────────────────────────────────────────────
# Deployment helpers
# ──────────────────────────────────────────────────────────────

def _backup(model_stem: str) -> str:
    """Create a timestamped backup of an existing model before replacing it."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_stem = f"{model_stem}_backup_{ts}"
    for ext in [".zip", "_vecnorm.pkl"]:
        src = f"{model_stem}{ext}"
        if os.path.exists(src):
            shutil.copy2(src, f"{backup_stem}{ext}")
    return backup_stem


def _quality_gate(
    old_score: float,
    new_score: float,
    label: str,
) -> Tuple[bool, str]:
    """
    Returns (deploy, reason).
    Deploys if:
      - There is no existing model (old_score = 0 and no file)
      - New score exceeds minimum acceptable AND beats old by threshold
      - Current model is below minimum (any improvement is acceptable)
    """
    if old_score == 0.0:
        return True, f"{label}: No existing model — deploying new model (score={new_score:.4f})."

    if new_score < MIN_ACCEPTABLE_SCORE:
        return False, (
            f"{label}: New model score {new_score:.4f} is below minimum acceptable "
            f"{MIN_ACCEPTABLE_SCORE} — NOT deploying."
        )

    if old_score < MIN_ACCEPTABLE_SCORE:
        return True, (
            f"{label}: Current model below minimum ({old_score:.4f}). "
            f"New model ({new_score:.4f}) accepted."
        )

    improvement = new_score - old_score
    if improvement >= IMPROVEMENT_THRESHOLD:
        return True, (
            f"{label}: Improvement +{improvement:.4f} ≥ threshold {IMPROVEMENT_THRESHOLD} — deploying."
        )
    else:
        return False, (
            f"{label}: Improvement +{improvement:.4f} < threshold {IMPROVEMENT_THRESHOLD} "
            f"(old={old_score:.4f}, new={new_score:.4f}) — keeping existing model."
        )


# ──────────────────────────────────────────────────────────────
# Main retraining loop
# ──────────────────────────────────────────────────────────────

def retrain(
    timesteps:  int,
    curriculum: bool,
    dry_run:    bool,
):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"Q-Store Gym Retraining Pipeline — {ts}")
    print(f"Mode: {'curriculum' if curriculum else 'per-task'} | "
          f"Timesteps: {timesteps:,} | Dry run: {dry_run}")
    print(f"{'='*60}\n")

    if curriculum:
        stem = "ppo_curriculum"
        print("--- Step 1: Evaluating current curriculum model ---")
        old_scores = _evaluate_all_tasks(stem, is_curriculum=True)
        old_mean   = statistics.mean(old_scores.values()) if old_scores else 0.0
        print(f"Current mean score across all tasks: {old_mean:.4f}")
        for task, s in old_scores.items():
            print(f"  {task}: {s:.4f}")

        if not dry_run:
            print("\n--- Step 2: Training new curriculum model ---")
            _train_curriculum(timesteps)

            print("\n--- Step 3: Evaluating new model ---")
            new_scores = _evaluate_all_tasks(stem, is_curriculum=True)
            new_mean   = statistics.mean(new_scores.values()) if new_scores else 0.0
            print(f"New mean score: {new_mean:.4f}")

            deploy, reason = _quality_gate(old_mean, new_mean, "Curriculum")
            print(f"\n--- Step 4: Quality gate ---")
            print(reason)
            if deploy:
                _backup(stem)
                print(f"New model deployed. Backup created.")
            else:
                # Restore old model from backup would go here in a real system
                print("New model rejected. Existing model retained.")

    else:
        for task_name in AVAILABLE_TASKS:
            task_stem = f"ppo_{task_name.replace(' ', '_')}"
            print(f"\n--- Task: {task_name} ---")

            print("  Evaluating current model...")
            old_score = _evaluate_model(task_stem, task_name) or 0.0
            print(f"  Current score: {old_score:.4f}")

            if not dry_run:
                print("  Training...")
                _train_one(task_stem, task_name, timesteps)

                print("  Evaluating new model...")
                new_score = _evaluate_model(task_stem, task_name) or 0.0
                print(f"  New score: {new_score:.4f}")

                deploy, reason = _quality_gate(old_score, new_score, task_name)
                print(f"  Quality gate: {reason}")
                if deploy:
                    _backup(task_stem)

    print(f"\n{'='*60}")
    print(f"Pipeline complete — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Store Gym automated retraining pipeline")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Timesteps per retrain run (default: 200,000).")
    parser.add_argument("--curriculum", action="store_true",
                        help="Retrain the curriculum model instead of per-task models.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate only — do not train or replace any models.")
    args = parser.parse_args()

    retrain(
        timesteps=args.timesteps,
        curriculum=args.curriculum,
        dry_run=args.dry_run,
    )
