import os
from stable_baselines3 import PPO
from gym_wrapper import QStoreGymWrapper
from tasks import AVAILABLE_TASKS

def train_agent(task_name: str, total_timesteps: int = 50000):
    print(f"\n=========================================\nInitializing pure training environment for task: {task_name}...")
    env = QStoreGymWrapper(task_name=task_name)
    
    model_filename = f"ppo_{task_name.replace(' ', '_')}"
    
    if os.path.exists(f"{model_filename}.zip"):
        print(f"Loading existing model weights from {model_filename}.zip for continuous learning...")
        model = PPO.load(model_filename, env=env)
    else:
        print(f"No existing model found for {task_name}. Creating a new PPO model from scratch...")
        model = PPO("MlpPolicy", env, verbose=0)
    
    print(f"Beginning training for {total_timesteps} steps... This will take a moment.")
    model.learn(total_timesteps=total_timesteps)
    
    model.save(model_filename)
    print(f"Training complete! Model successfully saved to {model_filename}.zip")

if __name__ == "__main__":
    # We will train an isolated, specific agent for each of the 3 tasks
    for task_name in AVAILABLE_TASKS:
        model_filename = f"ppo_{task_name.replace(' ', '_')}.zip"
        # Force rewrite existing models since we optimized the wrapper
        train_agent(task_name=task_name, total_timesteps=100000)
    
    print("\nAll specific task agents have been trained!")

