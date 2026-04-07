"""
train_crisis.py — Specialist trainer for The Strawberry Crisis scenario.

Key differences from the generic train.py:
- Uses StrawberryCrisisWrapper with urgency-shaping reward instead of generic wrapper.
- Higher entropy coefficient (0.05 vs 0.01) to keep the policy exploratory longer
  and avoid the local minimum of "do nothing → everything rots" at step 10.
- Larger n_steps=1024 to ensure PPO sees at least one full expiry cliff per rollout.
- More parallel envs (8) to generate diverse expiry timing samples.
- Saves to the standard ppo_The_Strawberry_Crisis.zip / _vecnorm.pkl so the existing
  API, eval script, and dashboard all work without any changes.

Usage:
    python train_crisis.py --timesteps 400000 --device auto
"""
import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from crisis_wrapper import StrawberryCrisisWrapper
from gym_wrapper import QStoreGymWrapper


MODEL_STEM   = "ppo_The_Strawberry_Crisis"
VECNORM_PATH = f"{MODEL_STEM}_vecnorm.pkl"
CHECKPOINT_DIR = "checkpoints/The_Strawberry_Crisis"


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            print(f"CUDA detected → GPU: {torch.cuda.get_device_name()}")
            return "cuda"
        # Apple MPS
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("Apple MPS detected → training on Metal GPU.")
            return "mps"
        print("No GPU detected → CPU.")
        return "cpu"
    return requested


def train(total_timesteps: int, device: str, n_envs: int = 8):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    resolved_device = resolve_device(device)

    print(f"\n{'='*60}")
    print(f"[Crisis Trainer] Strawberry Crisis — fresh from scratch")
    print(f"  timesteps : {total_timesteps:,}")
    print(f"  device    : {resolved_device}")
    print(f"  n_envs    : {n_envs}")
    print(f"{'='*60}\n")

    # --- Training envs (with urgency shaping) ---
    train_env = make_vec_env(
        lambda: Monitor(StrawberryCrisisWrapper()),
        n_envs=n_envs,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Eval env (generic wrapper, no crisis-specific reward shaping) ---
    eval_env_raw = DummyVecEnv([lambda: Monitor(QStoreGymWrapper(task_name="The Strawberry Crisis"))])
    eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, training=False)

    # --- PPO with crisis-tuned hyperparameters ---
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device=resolved_device,
        learning_rate=2e-4,        # slightly lower LR for stability
        n_steps=1024,              # MUST cover full 24-step episodes across 8 envs
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,             # HIGH entropy — prevents policy collapse to "hold and let rot"
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),  # larger network for complex perishable dynamics
    )

    eval_freq = max(1000, total_timesteps // 40)  # evaluate ~40x during training
    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=f"./{MODEL_STEM}_best",
            log_path=f"./{MODEL_STEM}_logs",
            eval_freq=eval_freq,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(5000, total_timesteps // 10),
            save_path=CHECKPOINT_DIR,
            name_prefix=MODEL_STEM,
            verbose=0,
        ),
    ])

    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=True)

    # Save to standard paths so API + eval_accuracy.py work unchanged
    model.save(MODEL_STEM)
    train_env.save(VECNORM_PATH)
    print(f"\n✅ Saved → {MODEL_STEM}.zip | {VECNORM_PATH}")
    print("Run `python eval_accuracy.py --task \"The Strawberry Crisis\" --n-episodes 20` to benchmark score improvement.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specialist trainer for The Strawberry Crisis")
    parser.add_argument("--timesteps", type=int, default=400_000, help="Total training steps (default: 400k)")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'mps', or 'cpu'")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments (default: 8)")
    args = parser.parse_args()

    train(total_timesteps=args.timesteps, device=args.device, n_envs=args.n_envs)
