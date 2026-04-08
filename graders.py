from __future__ import annotations

import os
import statistics
from typing import Callable, Dict, List, Optional

from env import QStoreEnv
from models import ActionSpace, ObservationSpace
from tasks import AVAILABLE_TASKS


DEFAULT_SEEDS = [11, 23, 37]


def deterministic_baseline_policy(observation: ObservationSpace) -> ActionSpace:
    return ActionSpace(
        pricing={item.product_id: 1.3 for item in observation.inventory},
        sourcing={},
        waste_management={},
    )


def cautious_clearance_policy(observation: ObservationSpace) -> ActionSpace:
    pricing: Dict[str, float] = {}
    waste_management: Dict[str, int] = {}
    for item in observation.inventory:
        multiplier = 1.1 if item.time_to_expiry_steps <= 6 else 1.4
        pricing[item.product_id] = multiplier
        if item.time_to_expiry_steps <= 2 and item.quantity > 20:
            waste_management[item.product_id] = min(5, item.quantity // 4)
    return ActionSpace(pricing=pricing, sourcing={}, waste_management=waste_management)


def inventory_balancing_policy(observation: ObservationSpace) -> ActionSpace:
    pricing: Dict[str, float] = {}
    sourcing: Dict[str, int] = {}
    for item in observation.inventory:
        demand = observation.demand_index.get(item.product_id, 1.0)
        pending = observation.pending_orders.get(item.product_id, 0)
        pricing[item.product_id] = 1.2 if demand > 2.0 else 1.45
        if item.quantity + pending < 25 and demand > 1.0:
            sourcing[item.product_id] = 10
    return ActionSpace(pricing=pricing, sourcing=sourcing, waste_management={})


PolicyFn = Callable[[ObservationSpace], ActionSpace]


def run_episode(task_name: str, policy: PolicyFn, seed: int, verbose: bool = False) -> Dict[str, float]:
    env = QStoreEnv(seed=seed)
    observation = env.reset(task_name, seed=seed)
    done = False
    total_reward = 0.0
    final_score = 0.0
    steps = 0

    while not done:
        action = policy(observation)
        result = env.step(action, verbose=verbose)
        observation = result.observation
        done = result.done
        total_reward += float(result.reward)
        final_score = float(result.score)
        steps += 1

    return {
        "task_name": task_name,
        "seed": float(seed),
        "steps": float(steps),
        "reward": total_reward,
        "score": max(0.001, min(0.999, final_score)),
        "net_profit": float(env.total_net_profit),
        "waste_value": float(env.total_waste_value),
    }


def grade_task(task_name: str, policy: PolicyFn = deterministic_baseline_policy, seeds: Optional[List[int]] = None) -> Dict[str, object]:
    seeds = seeds or list(DEFAULT_SEEDS)
    episodes = [run_episode(task_name, policy=policy, seed=seed) for seed in seeds]
    scores = [float(episode["score"]) for episode in episodes]
    return {
        "task_name": task_name,
        "episodes": episodes,
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "score_range_valid": all(0.0 < score < 1.0 for score in scores),
    }


def grade_all_tasks(policy: PolicyFn = deterministic_baseline_policy, seeds: Optional[List[int]] = None) -> List[Dict[str, object]]:
    return [grade_task(task_name, policy=policy, seeds=seeds) for task_name in AVAILABLE_TASKS]


def validate_task_graders(minimum_tasks: int = 3, seeds: Optional[List[int]] = None) -> Dict[str, object]:
    graded_tasks = grade_all_tasks(seeds=seeds)
    return {
        "task_count": len(graded_tasks),
        "minimum_tasks_met": len(graded_tasks) >= minimum_tasks,
        "all_scores_in_range": all(task["score_range_valid"] for task in graded_tasks),
        "tasks": graded_tasks,
    }


def validate_baseline_reproducibility(seeds: Optional[List[int]] = None) -> Dict[str, object]:
    seeds = seeds or list(DEFAULT_SEEDS)
    first_pass = grade_all_tasks(policy=deterministic_baseline_policy, seeds=seeds)
    second_pass = grade_all_tasks(policy=deterministic_baseline_policy, seeds=seeds)

    comparable = []
    reproducible = True
    for left, right in zip(first_pass, second_pass):
        left_scores = [episode["score"] for episode in left["episodes"]]
        right_scores = [episode["score"] for episode in right["episodes"]]
        match = left_scores == right_scores
        reproducible = reproducible and match
        comparable.append(
            {
                "task_name": left["task_name"],
                "scores_first_pass": left_scores,
                "scores_second_pass": right_scores,
                "match": match,
            }
        )

    return {"reproducible": reproducible, "comparisons": comparable}


def locate_saved_model(task_name: str, curriculum: bool = False) -> Optional[str]:
    if curriculum:
        candidates = ["ppo_curriculum.zip", os.path.join("ppo_curriculum_best", "best_model.zip")]
    else:
        stem = task_name.replace(" ", "_")
        candidates = [
            f"ppo_{stem}.zip",
            os.path.join(f"ppo_{stem}_best", "best_model.zip"),
        ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def validate_rl_assets() -> Dict[str, object]:
    per_task_models = {
        task_name: locate_saved_model(task_name) for task_name in AVAILABLE_TASKS
    }
    vecnorm_files = {
        task_name: os.path.exists(f"ppo_{task_name.replace(' ', '_')}_vecnorm.pkl") for task_name in AVAILABLE_TASKS
    }
    return {
        "per_task_models": per_task_models,
        "curriculum_model": locate_saved_model("", curriculum=True),
        "vecnorm_files": vecnorm_files,
        "has_any_saved_model": any(path is not None for path in per_task_models.values()),
    }
