import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env import QStoreEnv
from models import ActionSpace

PRODUCTS = ["milk", "bread", "chips", "strawberries"]
WEATHER_MAP = {"sunny": 0, "rainy": 1, "stormy": 2, "cloudy": 3}

class QStoreGymWrapper(gym.Env):
    """
    A Gym Wrapper for QStoreEnv to interface with Stable-Baselines3.
     Converts the Pydantic structured spaces into flat generic Box spaces.
    """
    def __init__(self, task_name="The Night Shift"):
        super().__init__()
        self.env = QStoreEnv()
        self.task_name = task_name
        
        # Action space: 4 products * 3 actions (pricing, sourcing, waste) = 12
        # Values between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # Observation space: 4 globals + 4 products * 5 attributes = 24 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_pydantic = self.env.reset(self.task_name)
        return self._flatten_obs(obs_pydantic), {}
        
    def step(self, action: np.ndarray):
        pricing = {}
        sourcing = {}
        waste_management = {}
        
        for i, p_id in enumerate(PRODUCTS):
            # Scale from [-1, 1] mapped to tighter logical bounds to prevent massive early penalties
            p_val = action[i*3 + 0]
            s_val = action[i*3 + 1]
            w_val = action[i*3 + 2]
            
            pricing[p_id] = float((p_val + 1.0) / 2.0 * 4.0) + 0.5 # $0.5 to $4.5
            sourcing[p_id] = int((s_val + 1.0) / 2.0 * 15) # 0 to 15 (prevent massive over-sourcing leading to rot)
            waste_management[p_id] = int((w_val + 1.0) / 2.0 * 10) # 0 to 10
            
        action_pydantic = ActionSpace(
            pricing=pricing,
            sourcing=sourcing,
            waste_management=waste_management
        )
        
        # We pass verbose=False to keep training clean
        result = self.env.step(action_pydantic, verbose=False)
        
        # Notice we use the step 'reward' not the final normalized 'score'
        # NORMALIZATION: Neural networks learn best when steps rewards are roughly between -1 and 1
        reward = float(result.reward) / 100.0 

        terminated = bool(result.done)
        truncated = False
        
        obs_arr = self._flatten_obs(result.observation)
        info = result.info
        info["score"] = float(result.score)
        
        return obs_arr, reward, terminated, truncated, info
        
    def _flatten_obs(self, obs_model):
        arr = [
            float(obs_model.current_step),
            float(obs_model.available_riders),
            float(WEATHER_MAP.get(obs_model.current_weather, 0)),
            1.0 if obs_model.special_event_active else 0.0
        ]
        
        inventory_dict = {item.product_id: item for item in obs_model.inventory}
        
        for p_id in PRODUCTS:
            item = inventory_dict.get(p_id)
            if item:
                arr.extend([
                    float(item.quantity),
                    float(item.cost_price),
                    float(item.time_to_expiry_steps),
                    float(obs_model.competitor_prices.get(p_id, 0.0)),
                    float(obs_model.demand_index.get(p_id, 0.0))
                ])
            else:
                arr.extend([0.0, 0.0, 0.0,
                            float(obs_model.competitor_prices.get(p_id, 0.0)),
                            float(obs_model.demand_index.get(p_id, 0.0))])
                            
        return np.array(arr, dtype=np.float32)
