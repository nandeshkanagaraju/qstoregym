import os
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv

from env import QStoreEnv
from models import ActionSpace
from tasks import AVAILABLE_TASKS
from gym_wrapper import QStoreGymWrapper

load_dotenv()

def run_baseline(task_idx: int, use_gpt: bool = False, use_ppo: bool = False):
    task_name = AVAILABLE_TASKS[task_idx]
    
    if use_ppo:
        print(f"\n=========================================\nLoading local PPO agent for: {task_name}")
        from stable_baselines3 import PPO
        
        model_filename = f"ppo_{task_name.replace(' ', '_')}"
        if not os.path.exists(f"{model_filename}.zip"):
            print(f"ERROR: {model_filename}.zip not found! Run train.py to train this specific task.")
            return
            
        model = PPO.load(model_filename)
        wrapper = QStoreGymWrapper(task_name=task_name)
        obs, _ = wrapper.reset()
        
        done = False
        final_score = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapper.step(action)
            done = terminated or truncated
            final_score = info.get("score", 0.0)
            
        print(f"Finished {task_name}. Final Score: {final_score:.4f}")
        return final_score

    # Original execution logic
    env = QStoreEnv()
    obs = env.reset(task_name)
    print(f"\n=========================================\nStarting Task: {task_name}")
    
    done = False
    has_key = bool(os.environ.get("OPENAI_API_KEY")) and use_gpt
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy_key")) if has_key else None
    
    while not done:
        if has_key:
            prompt = f"You are managing a dark store. Current state: {obs.model_dump_json()}.\nProvide a JSON object with 3 dict keys: 'pricing' (dict of product_id string to continuous float price), 'sourcing' (dict of product_id to integer restock quantity), 'waste_management' (dict of product_id to integer discard quantity)."
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a reinforcement learning agent optimizing store revenue. You must output a valid JSON object only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                action_data = json.loads(response.choices[0].message.content)
                action = ActionSpace(**action_data)
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                action = ActionSpace(pricing={}, sourcing={}, waste_management={})
        else:
            # Deterministic baseline when no API key
            pricing = {}
            for item in obs.inventory:
                pricing[item.product_id] = round(item.cost_price * 1.3, 2)
            action = ActionSpace(pricing=pricing, sourcing={}, waste_management={})
            
        result = env.step(action)
        obs = result.observation
        done = result.done
        
    print(f"Finished {task_name}. Final Score: {result.score:.4f}")
    return result.score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Store Inference")
    parser.add_argument("--use-gpt", action="store_true", help="Use OpenAI GPT-4o API for inference (slow, costs money)")
    parser.add_argument("--use-ppo", action="store_true", help="Use local trained PPO model (fast, free)")
    args = parser.parse_args()
    
    if args.use_gpt and not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Falling back to deterministic baseline.")
        args.use_gpt = False
        
    # Default behavior if nothing explicitly requested: try PPO, fallback to deterministic
    if not args.use_gpt and not args.use_ppo:
        if os.path.exists("ppo_The_Night_Shift.zip"):
            print("Found locally trained models. Using task-specific PPO agents natively.")
            args.use_ppo = True
        else:
            print("No local models found. Running with deterministic baseline.")
            
    for i in range(len(AVAILABLE_TASKS)):
        run_baseline(i, use_gpt=args.use_gpt, use_ppo=args.use_ppo)
