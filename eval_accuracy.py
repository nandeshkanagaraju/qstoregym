import os
import statistics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env import QStoreEnv
from models import ActionSpace
from tasks import AVAILABLE_TASKS
from gym_wrapper import QStoreGymWrapper

def run_baseline(task_name: str, n_episodes: int = 1) -> float:
    scores = []
    for _ in range(n_episodes):
        env = QStoreEnv()
        obs = env.reset(task_name)
        done = False
        result = None
        while not done:
            pricing = {item.product_id: 1.3 for item in obs.inventory}
            action = ActionSpace(pricing=pricing, sourcing={}, waste_management={})
            result = env.step(action, verbose=False)
            obs = result.observation
            done = result.done
        scores.append(result.score)
    return statistics.mean(scores)

def run_trained_ppo(task_name: str, n_episodes: int = 1):
    model_path = f"ppo_{task_name.replace(' ', '_')}.zip"
    if not os.path.exists(model_path):
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
    return statistics.mean(scores)

if __name__ == "__main__":
    print("\nEvaluating Q-Store Agent Accuracy across 5 Scenarios...")
    print("-" * 65)
    print(f"{'Scenario / Task Name':<25} | {'Baseline':<10} | {'Trained PPO':<12} | {'Net Gain'}")
    print("-" * 65)

    for task in AVAILABLE_TASKS:
        baseline_score = run_baseline(task)
        trained_score = run_trained_ppo(task)
        
        base_str = f"{baseline_score:.3f}"
        
        if trained_score is not None:
            train_str = f"{trained_score:.3f}"
            gain = trained_score - baseline_score
            gain_str = f"+{gain:.3f}" if gain > 0 else f"{gain:.3f}"
        else:
            train_str = "Skipped"
            gain_str = "N/A"
            
        print(f"{task:<25} | {base_str:<10} | {train_str:<12} | {gain_str}")

    print("-" * 65)
    print("* Score scale: 1.0 equals perfect theoretical max net profit with 0 waste.\n")
