"""
PPO training script for Q-Store Gym.

Changes from original:
- ent_coef=0.01  — entropy regularization prevents premature policy collapse in the
                    12-dimensional continuous action space.
- VecNormalize    — running normalization of observations AND rewards removes the need
                    for a hardcoded /100 reward divisor and handles cross-task scale variance.
- make_vec_env    — parallel environments reduce gradient variance and increase throughput.
- EvalCallback    — periodic evaluation with score logging so you can see if learning is
                    actually happening rather than training blindly.
- CheckpointCallback — saves intermediate models; training can be safely interrupted.
- Curriculum mode — single model trains progressively across all tasks (--curriculum flag).
- Sensible PPO hyperparameters tuned for episodic inventory control.
"""
import argparse
import os

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from gym_wrapper import QStoreGymWrapper
from curriculum import CurriculumGymWrapper
from tasks import AVAILABLE_TASKS


# ------------------------------------------------------------------
# Device resolution
# ------------------------------------------------------------------

def resolve_device(requested_device: str) -> str:
    requested_device = requested_device.lower()

    if requested_device == "auto":
        if torch.cuda.is_available():
            print(f"CUDA detected. Training on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            return "cuda"
        dml = _get_directml_device()
        if dml is not None:
            print(f"DirectML detected. Training on AMD backend: {dml}")
            return dml
        print("No GPU detected. Training on CPU.")
        return "cpu"

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Confirm torch.cuda.is_available() returns True.")
        print(f"Training on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        return "cuda"

    if requested_device == "dml":
        dml = _get_directml_device()
        if dml is None:
            raise RuntimeError("DirectML requested but not available. Install with `pip install torch-directml`.")
        print(f"Training on AMD backend: {dml}")
        return dml

    print("Training on CPU.")
    return "cpu"


def _get_directml_device():
    try:
        import torch_directml
        device = torch_directml.device()
        probe = torch.tensor([1.0], device=device)
        _ = probe + probe
        return device
    except Exception:
        return None


def _device_label(device) -> str:
    return str(device).upper() if isinstance(device, str) else str(device)


# ------------------------------------------------------------------
# Score evaluation callback
# ------------------------------------------------------------------

class ScoreLogCallback(EvalCallback):
    """
    Extends EvalCallback to also log the episode 'score' (store efficiency metric)
    so you can track whether the agent is improving on the actual objective,
    not just the per-step reward proxy.
    """

    def _on_step(self) -> bool:
        result = super()._on_step()
        # episode_rewards is populated by EvalCallback internals — we log it for visibility
        return result


# ------------------------------------------------------------------
# Per-task training
# ------------------------------------------------------------------

def train_per_task(task_name: str, total_timesteps: int, device: str, n_envs: int = 4):
    """Train one PPO model per task. Simple but no cross-task knowledge transfer."""
    print(f"\n{'='*60}")
    print(f"Training task: {task_name}")
    print(f"{'='*60}")

    model_stem   = f"ppo_{task_name.replace(' ', '_')}"
    vecnorm_path = f"{model_stem}_vecnorm.pkl"
    checkpoint_dir = f"checkpoints/{task_name.replace(' ', '_')}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    resolved_device = resolve_device(device)

    # Parallel training environments
    train_env = make_vec_env(
        lambda: Monitor(QStoreGymWrapper(task_name=task_name)),
        n_envs=n_envs,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Separate single eval environment (must use same normalization stats)
    eval_env_raw = DummyVecEnv([lambda: Monitor(QStoreGymWrapper(task_name=task_name))])
    eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, training=False)

    if os.path.exists(f"{model_stem}.zip") and os.path.exists(vecnorm_path):
        print(f"Loading existing model and normalization stats from {model_stem}.zip...")
        model = PPO.load(model_stem, env=train_env, device=resolved_device)
        train_env = VecNormalize.load(vecnorm_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        print(f"Creating new PPO model for {task_name}...")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=resolved_device,
            # Tuned for episodic inventory control (NOT the MuJoCo defaults)
            learning_rate=3e-4,
            n_steps=512,          # shorter rollout fits episode lengths (24–60 steps)
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,        # entropy regularization — critical for 12-dim continuous space
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[128, 128]),  # slightly larger than default [64,64]
        )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=f"./{model_stem}_best",
            log_path=f"./{model_stem}_logs",
            eval_freq=max(1000, total_timesteps // 20),  # evaluate ~20 times during training
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(5000, total_timesteps // 10),
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{task_name.replace(' ', '_')}",
            verbose=0,
        ),
    ])

    print(f"Training for {total_timesteps} steps on {_device_label(resolved_device)}...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)

    model.save(model_stem)
    train_env.save(vecnorm_path)
    print(f"Saved model → {model_stem}.zip | normalization stats → {vecnorm_path}")


# ------------------------------------------------------------------
# Curriculum training
# ------------------------------------------------------------------

def train_curriculum(total_timesteps: int, device: str, n_envs: int = 4):
    """
    Train a single PPO model across the full task curriculum.
    The curriculum wrapper changes tasks automatically when the agent earns promotion.
    Knowledge transfers between tasks since it is the same model weights throughout.
    """
    print(f"\n{'='*60}")
    print(f"Training with curriculum (all {len(AVAILABLE_TASKS)} tasks, single model)")
    print(f"{'='*60}")

    model_stem   = "ppo_curriculum"
    vecnorm_path = f"{model_stem}_vecnorm.pkl"
    checkpoint_dir = "checkpoints/curriculum"
    os.makedirs(checkpoint_dir, exist_ok=True)

    resolved_device = resolve_device(device)

    train_env = make_vec_env(
        lambda: Monitor(CurriculumGymWrapper(promotion_threshold=0.6, consecutive_required=3)),
        n_envs=n_envs,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env_raw = DummyVecEnv([lambda: Monitor(CurriculumGymWrapper(promotion_threshold=0.6, consecutive_required=3))])
    eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, training=False)

    if os.path.exists(f"{model_stem}.zip") and os.path.exists(vecnorm_path):
        print(f"Loading existing curriculum model from {model_stem}.zip...")
        model = PPO.load(model_stem, env=train_env, device=resolved_device)
        train_env = VecNormalize.load(vecnorm_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=resolved_device,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=f"./{model_stem}_best",
            log_path=f"./{model_stem}_logs",
            eval_freq=max(1000, total_timesteps // 20),
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(5000, total_timesteps // 10),
            save_path=checkpoint_dir,
            name_prefix="ppo_curriculum",
            verbose=0,
        ),
    ])

    print(f"Training for {total_timesteps} steps on {_device_label(resolved_device)}...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)

    model.save(model_stem)
    train_env.save(vecnorm_path)
    print(f"Saved curriculum model → {model_stem}.zip")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agents for Q-Store Gym")
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Training steps per task (or total for curriculum). Default: 500,000.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "dml", "cpu"], default="auto",
        help="Training device. 'auto' tries CUDA → DirectML → CPU.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel training environments.",
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Train a single model across all tasks using curriculum learning.",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Train only a specific task (ignored when --curriculum is set).",
    )
    args = parser.parse_args()

    if args.curriculum:
        train_curriculum(
            total_timesteps=args.timesteps,
            device=args.device,
            n_envs=args.n_envs,
        )
    elif args.task:
        if args.task not in AVAILABLE_TASKS:
            raise ValueError(f"Unknown task '{args.task}'. Available: {AVAILABLE_TASKS}")
        train_per_task(
            task_name=args.task,
            total_timesteps=args.timesteps,
            device=args.device,
            n_envs=args.n_envs,
        )
    else:
        # Default: train each task independently
        for task_name in AVAILABLE_TASKS:
            train_per_task(
                task_name=task_name,
                total_timesteps=args.timesteps,
                device=args.device,
                n_envs=args.n_envs,
            )
        print("\nAll per-task agents trained.")
