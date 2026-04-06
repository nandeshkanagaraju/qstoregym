"""
Inference runner for Q-Store Gym.

Supports three agent modes:
  --use-ppo       : Load a trained PPO model (per-task or curriculum).
  --use-gpt       : Use OpenAI GPT-4o as a zero-shot heuristic policy (requires OPENAI_API_KEY).
  --benchmark     : Run all three modes on all tasks and print a comparison table.

Default behavior: tries PPO → falls back to deterministic 1.3x baseline.

Changes from original:
- PPO inference now loads VecNormalize stats so inputs are correctly normalized.
- Per-step breakdown is printed during PPO runs (same detail as non-PPO mode).
- Benchmark mode compares deterministic / GPT / PPO side-by-side.
- Curriculum model supported via --curriculum flag.
"""
import os
import json
import argparse
import statistics
from typing import Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from env import QStoreEnv
from models import ActionSpace
from tasks import AVAILABLE_TASKS
from gym_wrapper import QStoreGymWrapper

load_dotenv()


# ------------------------------------------------------------------
# PPO inference
# ------------------------------------------------------------------

def _load_ppo(model_stem: str, task_name: str):
    """Load PPO model and its VecNormalize stats. Returns (model, norm_env)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    if not os.path.exists(f"{model_stem}.zip"):
        return None, None

    vecnorm_path = f"{model_stem}_vecnorm.pkl"

    raw_env  = DummyVecEnv([lambda: Monitor(QStoreGymWrapper(task_name=task_name))])
    if os.path.exists(vecnorm_path):
        norm_env = VecNormalize.load(vecnorm_path, raw_env)
        norm_env.training  = False
        norm_env.norm_reward = False
    else:
        # No normalization stats — run without (will perform worse but won't crash)
        norm_env = raw_env

    model = PPO.load(model_stem, env=norm_env)
    return model, norm_env


def run_ppo(task_name: str, curriculum: bool = False, stochastic: bool = False) -> Optional[float]:
    if curriculum:
        model_stem = "ppo_curriculum"
    else:
        model_stem = f"ppo_{task_name.replace(' ', '_')}"

    model, norm_env = _load_ppo(model_stem, task_name)
    if model is None:
        print(f"  [PPO] Model {model_stem}.zip not found. Run train.py first.")
        return None

    print(f"\n{'='*60}")
    print(f"[PPO{' Curriculum' if curriculum else ''}] Task: {task_name}")
    print(f"{'='*60}")

    obs = norm_env.reset()
    done = False
    final_score = 0.0

    # Run with the underlying QStoreEnv verbose=True output by using a parallel raw env
    raw_env = QStoreEnv()
    raw_obs = raw_env.reset(task_name)

    while not done:
        action, _ = model.predict(obs, deterministic=not stochastic)
        obs, _, terminated, infos = norm_env.step(action)
        done = bool(terminated[0])

        # Replay same action on raw env for verbose output
        from gym_wrapper import QStoreGymWrapper, PRODUCTS, PRICE_MULT_MIN, PRICE_MULT_MAX, MAX_SOURCE_UNITS, MAX_WASTE_UNITS
        act_arr = action[0]
        pricing, sourcing, waste = {}, {}, {}
        for i, p_id in enumerate(PRODUCTS):
            p_val = float(act_arr[i * 3])
            s_val = float(act_arr[i * 3 + 1])
            w_val = float(act_arr[i * 3 + 2])
            mult = PRICE_MULT_MIN + (p_val + 1.0) / 2.0 * (PRICE_MULT_MAX - PRICE_MULT_MIN)
            pricing[p_id]  = float(max(PRICE_MULT_MIN, min(PRICE_MULT_MAX, mult)))
            sourcing[p_id] = int(max(0.0, s_val) * MAX_SOURCE_UNITS)
            waste[p_id]    = int(max(0.0, w_val) * MAX_WASTE_UNITS)

        raw_result = raw_env.step(ActionSpace(pricing=pricing, sourcing=sourcing, waste_management=waste), verbose=True)
        final_score = float(infos[0].get("score", raw_result.score))
        raw_obs = raw_result.observation

    print(f"\n[Result] Task '{task_name}' final score: {final_score:.4f}")
    norm_env.close()
    return final_score


# ------------------------------------------------------------------
# GPT-4o inference
# ------------------------------------------------------------------

def run_gpt(task_name: str) -> Optional[float]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"  [GPT] OPENAI_API_KEY not set — skipping task '{task_name}'.")
        return None

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print(f"\n{'='*60}")
    print(f"[GPT-4o] Task: {task_name}")
    print(f"{'='*60}")

    env = QStoreEnv()
    obs = env.reset(task_name)
    done = False
    result = None

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False,
    )
    def _call_gpt(current_obs):
        prompt = (
            f"You are managing a dark store (Q-Commerce warehouse). "
            f"Current state: {current_obs.model_dump_json()}.\n\n"
            f"Output a JSON object with 3 keys:\n"
            f"  'pricing': dict of product_id -> price multiplier over cost_price "
            f"(e.g. 1.5 means price at 150%% of cost, range 0.8 to 3.0)\n"
            f"  'sourcing': dict of product_id -> integer units to order (0 to 20)\n"
            f"  'waste_management': dict of product_id -> integer units to discard (0 to 10)\n\n"
            f"Optimise for maximum net profit while minimising waste."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a store operations optimizer. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=15,  # hard timeout per request — never block indefinitely
        )
        return json.loads(response.choices[0].message.content)

    while not done:
        try:
            # Retries up to 3 times with exponential backoff before giving up
            action_data = _call_gpt(obs)
            action = ActionSpace(**action_data)
        except Exception as e:
            print(f"  [GPT] API failed after retries: {e}. Using deterministic fallback for this step.")
            pricing = {item.product_id: 1.5 for item in obs.inventory}
            action = ActionSpace(pricing=pricing, sourcing={}, waste_management={})

        result = env.step(action, verbose=True)
        obs = result.observation
        done = result.done

    score = result.score if result else 0.0
    print(f"\n[Result] Task '{task_name}' final score: {score:.4f}")
    return score


# ------------------------------------------------------------------
# Deterministic baseline
# ------------------------------------------------------------------

def run_deterministic(task_name: str) -> float:
    """Fixed 1.3x markup, no sourcing, no discarding. Useful as a lower-bound benchmark."""
    print(f"\n{'='*60}")
    print(f"[Deterministic Baseline] Task: {task_name}")
    print(f"{'='*60}")

    env = QStoreEnv()
    obs = env.reset(task_name)
    done = False
    result = None

    while not done:
        # 1.3x markup as a multiplier (ActionSpace.pricing is now a multiplier)
        pricing = {item.product_id: 1.3 for item in obs.inventory}
        action = ActionSpace(pricing=pricing, sourcing={}, waste_management={})
        result = env.step(action, verbose=True)
        obs = result.observation
        done = result.done

    score = result.score if result else 0.0
    print(f"\n[Result] Task '{task_name}' final score: {score:.4f}")
    return score


# ------------------------------------------------------------------
# Benchmark suite
# ------------------------------------------------------------------

def run_benchmark(n_episodes: int = 5, curriculum: bool = False):
    """
    Run deterministic / GPT / PPO on all tasks, n_episodes each.
    Prints a comparison table with mean ± std scores.
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {n_episodes} episodes per task per agent")
    print(f"{'='*60}")

    results = {}
    for task_name in AVAILABLE_TASKS:
        results[task_name] = {"deterministic": [], "ppo": []}

        for _ in range(n_episodes):
            results[task_name]["deterministic"].append(run_deterministic(task_name))

        for _ in range(n_episodes):
            score = run_ppo(task_name, curriculum=curriculum, stochastic=False)
            if score is not None:
                results[task_name]["ppo"].append(score)

        if os.environ.get("OPENAI_API_KEY"):
            results[task_name]["gpt"] = []
            for _ in range(n_episodes):
                score = run_gpt(task_name)
                if score is not None:
                    results[task_name]["gpt"].append(score)

    print(f"\n{'='*60}")
    print(f"{'Task':<28} {'Agent':<16} {'Mean Score':>10} {'Std':>8}")
    print(f"{'-'*62}")
    for task_name, agents in results.items():
        for agent_label, scores in agents.items():
            if scores:
                mean = statistics.mean(scores)
                std  = statistics.stdev(scores) if len(scores) > 1 else 0.0
                print(f"  {task_name:<26} {agent_label:<16} {mean:>10.4f} {std:>8.4f}")
    print(f"{'='*60}\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Store Gym Inference")
    parser.add_argument("--use-gpt",      action="store_true", help="Use GPT-4o for inference")
    parser.add_argument("--use-ppo",      action="store_true", help="Use trained PPO model")
    parser.add_argument("--curriculum",   action="store_true", help="Use curriculum PPO model")
    parser.add_argument("--stochastic",   action="store_true", help="Stochastic (non-deterministic) PPO actions")
    parser.add_argument("--benchmark",    action="store_true", help="Run full benchmark comparison")
    parser.add_argument("--n-episodes",   type=int, default=3, help="Episodes per task in benchmark mode")
    parser.add_argument("--task",         type=str, default=None, help="Run a specific task only")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and args.use_gpt:
        print("WARNING: OPENAI_API_KEY not set. Cannot use GPT mode.")
        args.use_gpt = False

    tasks = [args.task] if args.task and args.task in AVAILABLE_TASKS else AVAILABLE_TASKS

    if args.benchmark:
        run_benchmark(n_episodes=args.n_episodes, curriculum=args.curriculum)
    elif args.use_gpt:
        for t in tasks:
            run_gpt(t)
    elif args.use_ppo or args.curriculum:
        for t in tasks:
            run_ppo(t, curriculum=args.curriculum, stochastic=args.stochastic)
    else:
        # Auto-detect: use PPO if model exists, otherwise deterministic baseline
        has_ppo = any(
            os.path.exists(f"ppo_{t.replace(' ', '_')}.zip") for t in tasks
        ) or os.path.exists("ppo_curriculum.zip")

        if has_ppo:
            print("Found trained models. Running PPO inference.")
            for t in tasks:
                run_ppo(t, curriculum=os.path.exists("ppo_curriculum.zip"))
        else:
            print("No trained models found. Running deterministic baseline.")
            for t in tasks:
                run_deterministic(t)
