from pydantic import BaseModel, Field
from typing import Dict, List, Literal

class InventoryItem(BaseModel):
    product_id: str
    quantity: int
    cost_price: float
    time_to_expiry_steps: int

class ObservationSpace(BaseModel):
    current_step: int
    inventory: List[InventoryItem]
    competitor_prices: Dict[str, float]
    demand_index: Dict[str, float]
    available_riders: int
    current_weather: Literal['sunny', 'rainy', 'stormy', 'cloudy']
    special_event_active: bool
    # Units currently in transit per product (placed but not yet arrived).
    # Critical for rational sourcing decisions — agent must see what is already on order.
    pending_orders: Dict[str, int] = Field(default_factory=dict)

class ActionSpace(BaseModel):
    # Mapping product_id to a price MULTIPLIER over cost_price (e.g. 1.5 = 50% markup).
    # Using multipliers instead of absolute prices makes the action space scale-invariant
    # across products with different cost bases (chips $1 vs strawberries $4).
    pricing: Dict[str, float] = Field(default_factory=dict, description="Price multipliers over cost_price per product")
    # Mapping product_id to quantity ordered (arrives after lead time)
    sourcing: Dict[str, int] = Field(default_factory=dict, description="Inventory sourcing quantities")
    # Mapping product_id to quantity to discard (discards oldest batches first)
    waste_management: Dict[str, int] = Field(default_factory=dict, description="Discard quantities for items")

class RewardState(BaseModel):
    successful_sale_reward: float = 0.0
    efficiency_bonus: float = 0.0
    trust_penalty: float = 0.0
    waste_penalty: float = 0.0
    overhead_penalty: float = 0.0
    logistics_penalty: float = 0.0

    @property
    def total_reward(self) -> float:
        return (self.successful_sale_reward + self.efficiency_bonus) - (
            self.trust_penalty + self.waste_penalty + self.overhead_penalty + self.logistics_penalty
        )

class StepResult(BaseModel):
    observation: ObservationSpace
    reward: float
    reward_breakdown: RewardState
    done: bool
    score: float  # 0.0 to 1.0 (Store Efficiency Score), meaningful only at episode end
    info: Dict = Field(default_factory=dict)
