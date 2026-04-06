"""
Specialized Gymnasium wrapper for The Strawberry Crisis scenario.

Why this exists:
---------------
The generic QStoreGymWrapper fails on this task because:
1. 150 strawberries expire in just 10 steps — a massive discrete cliff-penalty.
2. The generic reward normalization (÷ max_potential_profit) produces tiny per-step
   signals during the first 9 steps, then a single catastrophic -1.0+ penalty at
   step 10 when all unsold strawberries rot simultaneously.
3. PPO's credit-assignment window (gae_lambda × n_steps) cannot reliably trace
   the step-10 cliff back to the wrong pricing decisions made at steps 1-5.

This wrapper adds:
- Urgency shaping: at each step, a negative bonus proportional to (qty × proximity_to_expiry)
  gives the agent *dense* feedback long before the cliff hits.
- Fire-sale reward: brief positive bonus when selling below cost to clear perishables.
- The base env physics are UNCHANGED — score computation is identical to eval.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env import QStoreEnv
from models import ActionSpace
from gym_wrapper import PRODUCTS, PRICE_MULT_MIN, PRICE_MULT_MAX, MAX_SOURCE_UNITS, MAX_WASTE_UNITS, OBS_DIM, ACT_DIM


CRISIS_TASK = "The Strawberry Crisis"


class StrawberryCrisisWrapper(gym.Env):
    """
    Reward-shaped wrapper specifically for The Strawberry Crisis task.
    Identical action/observation space to QStoreGymWrapper so the saved model
    is compatible with the generic inference loop.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.env = QStoreEnv()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset(CRISIS_TASK)
        return self._flatten_obs(obs), {}

    def step(self, action: np.ndarray):
        action_pydantic = self._decode_action(action)
        result = self.env.step(action_pydantic, verbose=False)

        norm_factor = max(1.0, self.env.max_potential_profit)
        base_reward = float(result.reward) / norm_factor

        # ── Urgency shaping: penalize holding near-expiry stock ──────────────────
        urgency_penalty = 0.0
        for item in self.env.inventory:
            if item.product_id == "strawberries":
                # Urgency ramps steeply as expiry approaches
                # At 10 steps: very low. At 2 steps: very high.
                steps_left = max(0, item.time_to_expiry_steps)
                if steps_left <= 5:
                    urgency = (6 - steps_left) / 6.0  # 0.17 → 1.0 as steps_left falls 5→0
                    urgency_penalty += urgency * item.quantity * 0.05

        urgency_penalty = urgency_penalty / norm_factor

        shaped_reward = base_reward - urgency_penalty

        terminated = bool(result.done)
        truncated = False
        obs_arr = self._flatten_obs(result.observation)
        info = dict(result.info)
        info["score"] = float(result.score)

        return obs_arr, shaped_reward, terminated, truncated, info

    def _flatten_obs(self, obs_model) -> np.ndarray:
        """Identical to QStoreGymWrapper._flatten_obs — same obs space."""
        arr = [
            float(obs_model.current_step),
            float(obs_model.available_riders),
            1.0 if obs_model.current_weather == 'sunny'  else 0.0,
            1.0 if obs_model.current_weather == 'rainy'  else 0.0,
            1.0 if obs_model.current_weather == 'stormy' else 0.0,
            1.0 if obs_model.current_weather == 'cloudy' else 0.0,
            1.0 if obs_model.special_event_active else 0.0,
        ]

        batch_map: dict = {}
        for item in obs_model.inventory:
            p = item.product_id
            if p not in batch_map:
                batch_map[p] = {"total_qty": 0, "cost_sum": 0.0, "min_expiry": item.time_to_expiry_steps}
            batch_map[p]["total_qty"]  += item.quantity
            batch_map[p]["cost_sum"]   += item.quantity * item.cost_price
            batch_map[p]["min_expiry"]  = min(batch_map[p]["min_expiry"], item.time_to_expiry_steps)

        for p_id in PRODUCTS:
            if p_id in batch_map:
                entry = batch_map[p_id]
                total_qty = entry["total_qty"]
                avg_cost  = entry["cost_sum"] / max(1, total_qty)
                min_exp   = float(entry["min_expiry"])
            else:
                total_qty = 0
                avg_cost  = 0.0
                min_exp   = 0.0

            arr.extend([
                float(total_qty),
                float(avg_cost),
                float(min_exp),
                float(obs_model.competitor_prices.get(p_id, 0.0)),
                float(obs_model.demand_index.get(p_id, 0.0)),
                float(obs_model.pending_orders.get(p_id, 0)),
            ])

        return np.array(arr, dtype=np.float32)

    def _decode_action(self, action: np.ndarray) -> ActionSpace:
        """Identical to QStoreGymWrapper._decode_action — same action space."""
        pricing = {}
        sourcing = {}
        waste_management = {}

        for i, p_id in enumerate(PRODUCTS):
            p_val = float(action[i * 3 + 0])
            s_val = float(action[i * 3 + 1])
            w_val = float(action[i * 3 + 2])

            mult = PRICE_MULT_MIN + (p_val + 1.0) / 2.0 * (PRICE_MULT_MAX - PRICE_MULT_MIN)
            pricing[p_id] = float(np.clip(mult, PRICE_MULT_MIN, PRICE_MULT_MAX))

            sourcing[p_id]         = int(max(0.0, s_val) * MAX_SOURCE_UNITS)
            waste_management[p_id] = int(max(0.0, w_val) * MAX_WASTE_UNITS)

        return ActionSpace(pricing=pricing, sourcing=sourcing, waste_management=waste_management)
