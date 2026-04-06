import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env import QStoreEnv
from models import ActionSpace

# All 4 products. Tasks with fewer products get zeros in unused slots.
# Consistent shape means one model can train/infer across all tasks.
PRODUCTS = ["milk", "bread", "chips", "strawberries"]

# Pricing multiplier bounds: agent sets price as a multiple of cost_price.
# 0.8x = controlled fire sale, 3.0x = premium pricing.
# Centering the [-1,1] range here puts the default (action=0) at 1.9x — a profitable default.
PRICE_MULT_MIN = 0.8
PRICE_MULT_MAX = 3.0

# Max sourcing per step and max manual discard per step.
# Sourcing/waste use half-rectified encoding: action ≤ 0 → 0 units (do nothing).
# This eliminates the cold-start penalty hole where a zero-initialized network
# would immediately order ~7 units and discard ~5 units per product per step.
MAX_SOURCE_UNITS = 20
MAX_WASTE_UNITS = 10

# Observation layout (31 dimensions total):
#   [0]     current_step
#   [1]     available_riders
#   [2]     weather_sunny   (one-hot)
#   [3]     weather_rainy   (one-hot)
#   [4]     weather_stormy  (one-hot)
#   [5]     weather_cloudy  (one-hot)
#   [6]     special_event_active
#   Per product (×4, starting at index 7): 6 dims each = 24 dims
#     [+0]  total_quantity   (sum across all batches)
#     [+1]  weighted_avg_cost
#     [+2]  min_time_to_expiry (most urgent batch)
#     [+3]  competitor_price
#     [+4]  demand_index
#     [+5]  pending_qty (units already ordered but not yet arrived)
OBS_DIM = 7 + len(PRODUCTS) * 6  # = 31

# Action layout (12 dimensions):
#   Per product (×4, starting at index 0): 3 dims each = 12 dims
#     [+0]  pricing action  → maps [-1,1] to [PRICE_MULT_MIN, PRICE_MULT_MAX]
#     [+1]  sourcing action → max(0, val) * MAX_SOURCE_UNITS  (half-rectified)
#     [+2]  waste action    → max(0, val) * MAX_WASTE_UNITS   (half-rectified)
ACT_DIM = len(PRODUCTS) * 3  # = 12


class QStoreGymWrapper(gym.Env):
    """
    Gymnasium wrapper for QStoreEnv. Converts Pydantic structured spaces to flat
    numpy arrays compatible with Stable-Baselines3.

    Key design decisions vs the original wrapper:
    - Observation aggregates ALL inventory batches per product (not just the last one).
    - Pending orders are included in the observation so the agent can avoid over-sourcing.
    - Weather is one-hot encoded (4 dims) instead of ordinal (1 dim).
    - Pricing outputs a cost multiplier, not an absolute dollar price, so the action
      space scales correctly for products with very different cost bases.
    - Sourcing and waste use half-rectified encoding so a zero-initialized network
      produces zero-action behavior (no spurious early penalties).
    """

    metadata = {"render_modes": []}

    def __init__(self, task_name: str = "The Night Shift"):
        super().__init__()
        self.env = QStoreEnv()
        self.task_name = task_name

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_pydantic = self.env.reset(self.task_name)
        return self._flatten_obs(obs_pydantic), {}

    def step(self, action: np.ndarray):
        action_pydantic = self._decode_action(action)
        result = self.env.step(action_pydantic, verbose=False)

        # Normalize reward per-step. We divide by max_potential_profit (a stable,
        # task-specific constant) to produce a roughly unit-scale signal rather than
        # an arbitrary /100 that breaks on different task sizes.
        norm_factor = max(1.0, self.env.max_potential_profit)
        reward = float(result.reward) / norm_factor

        terminated = bool(result.done)
        truncated = False
        obs_arr = self._flatten_obs(result.observation)
        info = dict(result.info)
        info["score"] = float(result.score)

        return obs_arr, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation flattening
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs_model) -> np.ndarray:
        arr = [
            float(obs_model.current_step),
            float(obs_model.available_riders),
            # One-hot weather (fixes ordinal encoding bug)
            1.0 if obs_model.current_weather == 'sunny'  else 0.0,
            1.0 if obs_model.current_weather == 'rainy'  else 0.0,
            1.0 if obs_model.current_weather == 'stormy' else 0.0,
            1.0 if obs_model.current_weather == 'cloudy' else 0.0,
            1.0 if obs_model.special_event_active else 0.0,
        ]

        # Aggregate ALL batches per product before encoding.
        # Original code kept only the last batch, silently discarding earlier batches.
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
                float(obs_model.pending_orders.get(p_id, 0)),  # in-transit visibility
            ])

        return np.array(arr, dtype=np.float32)

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_action(self, action: np.ndarray) -> ActionSpace:
        """
        Decode the flat [-1, 1] action vector into a structured ActionSpace.

        Pricing:  [-1, 1] → [PRICE_MULT_MIN, PRICE_MULT_MAX] (linear, full range used)
        Sourcing: max(0, val) * MAX_SOURCE_UNITS  — negative = do nothing (cold-start safe)
        Waste:    max(0, val) * MAX_WASTE_UNITS   — negative = do nothing (cold-start safe)
        """
        pricing = {}
        sourcing = {}
        waste_management = {}

        for i, p_id in enumerate(PRODUCTS):
            p_val = float(action[i * 3 + 0])
            s_val = float(action[i * 3 + 1])
            w_val = float(action[i * 3 + 2])

            # Pricing multiplier: linearly spans [PRICE_MULT_MIN, PRICE_MULT_MAX]
            mult = PRICE_MULT_MIN + (p_val + 1.0) / 2.0 * (PRICE_MULT_MAX - PRICE_MULT_MIN)
            pricing[p_id] = float(np.clip(mult, PRICE_MULT_MIN, PRICE_MULT_MAX))

            # Half-rectified: action ≤ 0 → 0 units ordered/discarded
            sourcing[p_id]          = int(max(0.0, s_val) * MAX_SOURCE_UNITS)
            waste_management[p_id]  = int(max(0.0, w_val) * MAX_WASTE_UNITS)

        return ActionSpace(pricing=pricing, sourcing=sourcing, waste_management=waste_management)
