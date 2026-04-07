import statistics
import argparse
import os
from typing import List, Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from graders import DEFAULT_SEEDS, deterministic_baseline_policy, locate_saved_model, run_episode
from tasks import AVAILABLE_TASKS
from gym_wrapper import QStoreGymWrapper


DEFAULT_EVAL_EPISODES = 20


def _summarize(scores: List[float]) -> Tuple[float, float]:
    mean = statistics.mean(scores) if scores else 0.0
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return mean, std


def run_baseline(task_name: str, n_episodes: int = DEFAULT_EVAL_EPISODES) -> List[float]:
    seeds = [DEFAULT_SEEDS[0] + idx for idx in range(n_episodes)]
    return [float(run_episode(task_name, policy=deterministic_baseline_policy, seed=seed)["score"]) for seed in seeds]


def run_trained_ppo(task_name: str, n_episodes: int = DEFAULT_EVAL_EPISODES) -> Optional[List[float]]:
    model_path = locate_saved_model(task_name)
    if model_path is None or not os.path.exists(model_path):
        return None

    raw_env_func = lambda: Monitor(QStoreGymWrapper(task_name=task_name))
    raw_env = DummyVecEnv([raw_env_func])

    vecnorm_path = f"ppo_{task_name.replace(' ', '_')}_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        norm_env = VecNormalize.load(vecnorm_path, raw_env)
        norm_env.training = False
        norm_env.norm_reward = False
    else:
        norm_env = raw_env

    try:
        model = PPO.load(model_path, env=norm_env)
    except Exception:
        norm_env.close()
        return None

    scores = []
    for _ in range(n_episodes):
        obs = norm_env.reset()
        done = False
        score = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, infos = norm_env.step(action)
            done = bool(terminated[0])
            score = float(infos[0].get("score", 0.0))
        scores.append(score)

    norm_env.close()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Q-Store score performance")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Evaluation episodes per scenario (default: {DEFAULT_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--task",
        choices=AVAILABLE_TASKS,
        default=None,
        help="Evaluate a single task instead of all scenarios",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else AVAILABLE_TASKS

    print(f"\nBenchmarking Q-Store score performance across {len(tasks)} scenario(s)...")
    print(f"Using {args.n_episodes} evaluation episode(s) per scenario.")
    print("-" * 115)
    print(
        f"{'Scenario / Task Name':<25} | "
        f"{'Baseline Mean':>13} | {'Baseline Std':>12} | "
        f"{'Trained Mean':>12} | {'Trained Std':>11} | {'Net Gain':>9}"
    )
    print("-" * 115)

    for task in tasks:
        baseline_scores = run_baseline(task, n_episodes=args.n_episodes)
        baseline_mean, baseline_std = _summarize(baseline_scores)
        trained_scores = run_trained_ppo(task, n_episodes=args.n_episodes)

        base_mean_str = f"{baseline_mean:.3f}"
        base_std_str = f"{baseline_std:.3f}"

        if trained_scores is not None:
            trained_mean, trained_std = _summarize(trained_scores)
            trained_mean_str = f"{trained_mean:.3f}"
            trained_std_str = f"{trained_std:.3f}"
            gain = trained_mean - baseline_mean
            gain_str = f"+{gain:.3f}" if gain > 0 else f"{gain:.3f}"
        else:
            trained_mean_str = "Skipped"
            trained_std_str = "Skipped"
            gain_str = "N/A"

        print(
            f"{task:<25} | "
            f"{base_mean_str:>13} | {base_std_str:>12} | "
            f"{trained_mean_str:>12} | {trained_std_str:>11} | {gain_str:>9}"
        )

    print("-" * 115)
    print("* Score scale: 1.0 means matching the initial-inventory profit ceiling with zero waste.\n")
