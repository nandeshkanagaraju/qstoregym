import random
import math
from typing import Dict, List, Literal, Tuple
from models import ObservationSpace, ActionSpace, RewardState, StepResult, InventoryItem
from tasks import get_task_config

# Assume 1 step = 15 minutes.
class QStoreEnv:
    def __init__(self):
        self.task_name = ""
        self.current_step = 0
        self.max_steps = 0
        self.inventory: List[InventoryItem] = []
        self.competitor_prices: Dict[str, float] = {}
        self.demand_index: Dict[str, float] = {}
        self.available_riders = 0
        self.current_weather = 'sunny'
        self.special_event_active = False
        self.config = {}
        
        # Track pending orders: list of (arrival_step, product_id, quantity)
        self.pending_orders: List[Tuple[int, str, int]] = []
        
        # Metrics for score calculation
        self.total_net_profit = 0.0
        self.max_potential_profit = 1.0 # to avoid div by zero initially
        self.total_waste_value = 0.0
        self.total_inventory_cost_at_start = 0.0
        
    def reset(self, task_name: str = "The Night Shift") -> ObservationSpace:
        self.task_name = task_name
        self.config = get_task_config(task_name)
        
        self.current_step = 0
        self.max_steps = self.config["max_steps"]
        self.inventory = [InventoryItem(**item) for item in self.config["initial_inventory"]]
        self.available_riders = self.config["initial_riders"]
        
        self.weather_states: List[Literal['sunny', 'rainy', 'stormy', 'cloudy']] = ['sunny', 'rainy', 'stormy', 'cloudy']
        self.weather_probs = [self.config["weather_prob"][w] for w in self.weather_states]
        
        self.current_weather = random.choices(self.weather_states, weights=self.weather_probs, k=1)[0]
        self.special_event_active = random.random() < self.config["special_event_prob"]
        
        self.pending_orders = []
        
        self.total_net_profit = 0.0
        self.total_waste_value = 0.0
        self.total_inventory_cost_at_start = sum([item.quantity * item.cost_price for item in self.inventory])
        # approximate max potential if everything sold at 1.5x cost
        self.max_potential_profit = sum([item.quantity * item.cost_price * 1.5 for item in self.inventory]) + 0.01 
        
        self._update_environment_dynamics()
        
        return self.state()
        
    def state(self) -> ObservationSpace:
        return ObservationSpace(
            current_step=self.current_step,
            inventory=self.inventory,
            competitor_prices=self.competitor_prices,
            demand_index=self.demand_index,
            available_riders=self.available_riders,
            current_weather=self.current_weather,
            special_event_active=self.special_event_active
        )
        
    def _update_environment_dynamics(self):
        # Update weather randomly
        self.current_weather = random.choices(self.weather_states, weights=self.weather_probs, k=1)[0]
        self.special_event_active = random.random() < self.config["special_event_prob"]
        
        # Base competitor prices
        products = set([item.product_id for item in self.inventory])
        for p in products:
            cost = next((item.cost_price for item in self.inventory if item.product_id == p), 1.0)
            self.competitor_prices[p] = round(cost * random.uniform(1.2, 1.8), 2)
            
            # Demand multiplier based on weather and events
            weather_mult = 1.5 if self.current_weather in ['rainy', 'stormy'] else 1.0
            event_mult = 2.0 if self.special_event_active else 1.0
            
            self.demand_index[p] = self.config["base_demand"] * weather_mult * event_mult * random.uniform(0.8, 1.2)
            
        # Rider fluctuations
        if self.current_weather == 'stormy':
            self.available_riders = max(1, self.available_riders - random.randint(1, 3))
        elif random.random() < 0.2:
            self.available_riders += random.choice([-1, 1])
            self.available_riders = max(1, self.available_riders)

    def step(self, action: ActionSpace, verbose: bool = True) -> StepResult:
        if verbose:
            print(f"\n--- [{self.task_name}] Step {self.current_step + 1}/{self.max_steps} ---")
            print(f"Weather: {self.current_weather} | Event Active: {self.special_event_active} | Available Riders: {self.available_riders}")

        
        reward_state = RewardState()
        
        # 1. Process Waste Management (manual discard)
        for p_id, discard_qty in action.waste_management.items():
            if discard_qty > 0:
                if verbose:
                    print(f"  [Waste] Agent manually discarded {discard_qty} units of {p_id}")
                self._discard_inventory(p_id, discard_qty, reward_state, manual=True)

        # 2. Process Sourcing (lead time = 4 steps = 1 hr)
        for p_id, order_qty in action.sourcing.items():
            if order_qty > 0:
                if verbose:
                    print(f"  [Sourcing] Agent ordered {order_qty} units of {p_id} (arrives in 4 steps)")
                cost = next((item.cost_price for item in self.inventory if item.product_id == p_id), 2.0)
                self.pending_orders.append((self.current_step + 4, p_id, order_qty))
                self.max_potential_profit += order_qty * cost * 0.5 # Add expected margin to potential profit
                
        # 3. Process Arrivals
        arrived_orders = [o for o in self.pending_orders if o[0] <= self.current_step]
        self.pending_orders = [o for o in self.pending_orders if o[0] > self.current_step]
        for _, p_id, qty in arrived_orders:
            if verbose:
                print(f"  [Arrival] {qty} units of {p_id} have arrived to inventory")
            cost = next((item.cost_price for item in self.inventory if item.product_id == p_id), 2.0)
            self.inventory.append(InventoryItem(product_id=p_id, quantity=qty, cost_price=cost, time_to_expiry_steps=100))
                
        # 4. Resolve Pricing and Sales
        total_orders = 0
        if verbose:
            print("  [Sales Report]")
        for p_id, agent_price in action.pricing.items():
            cost = next((item.cost_price for item in self.inventory if item.product_id == p_id), None)
            if not cost: continue
            
            comp_price = self.competitor_prices.get(p_id, agent_price * 1.5)
            demand = self.demand_index.get(p_id, 1.0)
            
            # Continuous pricing feedback. Too high price vs competitor = zero sales.
            price_ratio = comp_price / max(0.01, agent_price)
            expected_sales = int(demand * 10 * (price_ratio ** 2))
            
            # Execute sales 
            sales_achieved = self._sell_inventory(p_id, expected_sales, agent_price, comp_price, reward_state)
            total_orders += sales_achieved
            
            if verbose:
                print(f"    - {p_id}: Price=${agent_price:.2f} (Comp=${comp_price:.2f}), DemandIdx={demand:.2f}")
                print(f"      -> Expected Sales: {expected_sales}, Actual Sold: {sales_achieved}")
            
            if expected_sales > sales_achieved:
                # Trust penalty due to stockout
                reward_state.trust_penalty += (expected_sales - sales_achieved) * 0.5
                
        # 5. Logistics Check
        max_delivery_capacity = self.available_riders * 4 # Each rider handles 4 orders per step max
        if total_orders > max_delivery_capacity:
            missed_orders = total_orders - max_delivery_capacity
            if verbose:
                print(f"  [Logistics] Missed {missed_orders} orders due to rider limit! (Total orders: {total_orders}, Capacity: {max_delivery_capacity})")
            reward_state.logistics_penalty += missed_orders * 1.0 # 1 point per missed order
            
        # 6. Expiry and Overhead
        total_items = 0
        new_inventory = []
        for item in self.inventory:
            total_items += item.quantity
            item.time_to_expiry_steps -= 1
            if item.time_to_expiry_steps <= 0:
                if verbose:
                    print(f"  [Rot] {item.quantity} units of {item.product_id} expired on the shelf!")
                reward_state.waste_penalty += item.quantity * item.cost_price * 1.5 # 1.5x penalty rot
                self.total_waste_value += item.quantity * item.cost_price
            elif item.quantity > 0:
                new_inventory.append(item)
        
        self.inventory = new_inventory
        reward_state.overhead_penalty += total_items * 0.01 # Small storage fee
        
        # Advance Step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if not done:
            self._update_environment_dynamics()
        
        # Calculate Score
        waste_ratio = min(1.0, self.total_waste_value / max(1.0, (self.total_inventory_cost_at_start + self.total_waste_value)))
        profit_ratio = max(0.0, self.total_net_profit / self.max_potential_profit)
        score = max(0.0, profit_ratio - waste_ratio)
        if score > 1.0: score = 1.0
        
        if verbose:
            print(f"  [Score Calculation]")
            print(f"    - Net Profit: ${self.total_net_profit:.2f} / Max Potential: ${self.max_potential_profit:.2f}")
            print(f"    - Total Waste Value: ${self.total_waste_value:.2f} / Started With: ${self.total_inventory_cost_at_start:.2f}")
            print(f"    - Profit Ratio: {profit_ratio:.4f} | Waste Ratio: {waste_ratio:.4f} -> Final Running Score: {score:.4f}")
        
        return StepResult(
            observation=self.state(),
            reward=reward_state.total_reward,
            reward_breakdown=reward_state,
            done=done,
            score=score,
            info={"net_profit": self.total_net_profit, "waste_value": self.total_waste_value}
        )

    def _sell_inventory(self, p_id: str, expected_sales: int, price: float, comp_price: float, reward_state: RewardState) -> int:
        remaining_qty = expected_sales
        sales_made = 0
        
        p_items = [item for item in self.inventory if item.product_id == p_id]
        p_items.sort(key=lambda x: x.time_to_expiry_steps)
        
        for item in p_items:
            if remaining_qty <= 0: break
            
            sell_qty = min(item.quantity, remaining_qty)
            item.quantity -= sell_qty
            remaining_qty -= sell_qty
            sales_made += sell_qty
            
            profit_margin = (price - item.cost_price) * sell_qty
            self.total_net_profit += profit_margin
            
            if profit_margin > 0:
                reward_state.successful_sale_reward += profit_margin
            else:
                reward_state.waste_penalty += abs(profit_margin) # selling at loss
                
            if item.time_to_expiry_steps <= 4 and profit_margin >= 0:
                reward_state.efficiency_bonus += sell_qty * 0.5
                
        return sales_made
        
    def _discard_inventory(self, p_id: str, quantity: int, reward_state: RewardState, manual: bool):
        remaining_qty = quantity
        p_items = [item for item in self.inventory if item.product_id == p_id]
        p_items.sort(key=lambda x: x.time_to_expiry_steps)
        
        for item in p_items:
            if remaining_qty <= 0: break
            discard_qty = min(item.quantity, remaining_qty)
            item.quantity -= discard_qty
            remaining_qty -= discard_qty
            
            loss = discard_qty * item.cost_price
            self.total_waste_value += loss
            reward_state.waste_penalty += loss * (1.0 if manual else 1.5) 
