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

class ActionSpace(BaseModel):
    # Mapping product_id to a new continuous price value
    pricing: Dict[str, float] = Field(default_factory=dict, description="Dynamic pricing for items")
    # Mapping product_id to quantity ordered (arrives after lead time)
    sourcing: Dict[str, int] = Field(default_factory=dict, description="Inventory sourcing quantities")
    # Mapping product_id to quantity to discard (note: agent discards specific batches, but for simplicity, discards oldest first)
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
    score: float  # 0.0 to 1.0 (Store Efficiency Score)
    info: Dict = Field(default_factory=dict)
