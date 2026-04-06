import random
from typing import Dict, List, Literal, Tuple
from models import ObservationSpace, ActionSpace, RewardState, StepResult, InventoryItem
from tasks import get_task_config

# 1 step = 15 minutes of real store time.
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

        # Pending orders: list of (arrival_step, product_id, quantity)
        self.pending_orders: List[Tuple[int, str, int]] = []

        # Cumulative episode metrics for score calculation
        self.total_net_profit = 0.0
        self.total_waste_value = 0.0
        # Fixed at reset — NOT inflated by sourcing orders.
        # Represents the theoretical max revenue from initial inventory at 1.5x cost.
        # The denominator stays stable so profit_ratio is meaningful throughout the episode.
        self.max_potential_profit = 1.0

    def reset(self, task_name: str = "The Night Shift") -> ObservationSpace:
        self.task_name = task_name
        self.config = get_task_config(task_name)

        self.current_step = 0
        self.max_steps = self.config["max_steps"]
        self.inventory = [InventoryItem(**item) for item in self.config["initial_inventory"]]
        self.available_riders = self.config["initial_riders"]

        self.weather_states: List[Literal['sunny', 'rainy', 'stormy', 'cloudy']] = ['sunny', 'rainy', 'stormy', 'cloudy']
        self.weather_probs = [self.config["weather_prob"][w] for w in self.weather_states]

        self.pending_orders = []
        self.total_net_profit = 0.0
        self.total_waste_value = 0.0

        # Lock max_potential_profit at reset — covers only initial inventory.
        # Sourcing no longer inflates this denominator, removing the double-penalty bug.
        self.max_potential_profit = max(
            1.0,
            sum(item.quantity * item.cost_price * 1.5 for item in self.inventory)
        )

        # Single call to set weather/event and competitor prices.
        self._update_environment_dynamics()

        return self.state()

    def state(self) -> ObservationSpace:
        # Aggregate pending orders per product so the agent can see what is in transit.
        pending_summary: Dict[str, int] = {}
        for _, p_id, qty in self.pending_orders:
            pending_summary[p_id] = pending_summary.get(p_id, 0) + qty

        return ObservationSpace(
            current_step=self.current_step,
            inventory=self.inventory,
            competitor_prices=self.competitor_prices,
            demand_index=self.demand_index,
            available_riders=self.available_riders,
            current_weather=self.current_weather,
            special_event_active=self.special_event_active,
            pending_orders=pending_summary,
        )

    def _update_environment_dynamics(self):
        self.current_weather = random.choices(self.weather_states, weights=self.weather_probs, k=1)[0]
        self.special_event_active = random.random() < self.config["special_event_prob"]

        products = set(item.product_id for item in self.inventory)
        for p in products:
            cost = next((item.cost_price for item in self.inventory if item.product_id == p), 1.0)
            self.competitor_prices[p] = round(cost * random.uniform(1.2, 1.8), 2)

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

        # 2. Process Sourcing.
        # Only source products that exist in this task's product catalog to prevent phantom inventory.
        known_products = set(item.product_id for item in self.inventory) | set(self.config["product_expiry_steps"].keys())
        for p_id, order_qty in action.sourcing.items():
            if order_qty <= 0:
                continue
            if p_id not in known_products:
                # Silently ignore sourcing for products not in this task.
                continue
            if verbose:
                print(f"  [Sourcing] Agent ordered {order_qty} units of {p_id} (arrives in 4 steps)")
            self.pending_orders.append((self.current_step + 4, p_id, order_qty))

        # 3. Process Arrivals — items that have reached their arrival step.
        arrived_orders = [o for o in self.pending_orders if o[0] <= self.current_step]
        self.pending_orders = [o for o in self.pending_orders if o[0] > self.current_step]
        for _, p_id, qty in arrived_orders:
            if verbose:
                print(f"  [Arrival] {qty} units of {p_id} have arrived to inventory")
            cost = next((item.cost_price for item in self.inventory if item.product_id == p_id), None)
            if cost is None:
                # Product not currently stocked — use config cost if available.
                initial_items = {i["product_id"]: i["cost_price"] for i in self.config["initial_inventory"]}
                cost = initial_items.get(p_id, 2.0)

            # Use product-specific expiry from config (fixes the hardcoded 100-step bug).
            expiry = self.config["product_expiry_steps"].get(p_id, 48)
            self.inventory.append(InventoryItem(
                product_id=p_id,
                quantity=qty,
                cost_price=cost,
                time_to_expiry_steps=expiry,
            ))

        # 4. Resolve Pricing and Sales.
        # action.pricing contains MULTIPLIERS over cost_price, not absolute prices.
        max_delivery_capacity = self.available_riders * 4
        remaining_capacity = max_delivery_capacity
        total_missed_orders = 0

        if verbose:
            print("  [Sales Report]")
        for p_id, price_multiplier in action.pricing.items():
            cost = next((item.cost_price for item in self.inventory if item.product_id == p_id), None)
            if cost is None:
                continue

            agent_price = cost * max(0.5, price_multiplier)  # floor at 50% of cost (deep fire sale)
            comp_price = self.competitor_prices.get(p_id, agent_price * 1.5)
            demand = self.demand_index.get(p_id, 1.0)

            # Linear demand model with cap — prevents explosive underprice incentive.
            # price_factor in [0.1, 3.0]: agent at 50% of comp price gets 2x demand, at 200% gets 0.5x.
            price_factor = min(3.0, max(0.1, comp_price / max(0.01, agent_price)))
            expected_sales = int(demand * 10 * price_factor)

            # Rider capacity constraint
            fulfillable_sales = min(expected_sales, remaining_capacity)
            missed_logistics = expected_sales - fulfillable_sales
            total_missed_orders += missed_logistics
            remaining_capacity -= fulfillable_sales

            sales_achieved = self._sell_inventory(p_id, fulfillable_sales, agent_price, reward_state)

            if verbose:
                print(f"    - {p_id}: Price=${agent_price:.2f} (Mult={price_multiplier:.2f}x, Comp=${comp_price:.2f}), Demand={demand:.2f}")
                print(f"      -> Expected: {expected_sales}, Fulfilled: {sales_achieved}")

            stockout_misses = fulfillable_sales - sales_achieved
            if stockout_misses > 0:
                reward_state.trust_penalty += stockout_misses * 1.0

        # 5. Logistics penalty for rider-constrained missed orders.
        if total_missed_orders > 0:
            if verbose:
                print(f"  [Logistics] {total_missed_orders} orders missed due to rider capacity (max {max_delivery_capacity}).")
            reward_state.logistics_penalty += total_missed_orders * 1.5

        # 6. Expiry check and overhead.
        total_items = 0
        new_inventory = []
        for item in self.inventory:
            total_items += item.quantity
            item.time_to_expiry_steps -= 1
            if item.time_to_expiry_steps <= 0:
                if verbose:
                    print(f"  [Rot] {item.quantity} units of {item.product_id} expired!")
                reward_state.waste_penalty += item.quantity * item.cost_price * 1.5
                self.total_waste_value += item.quantity * item.cost_price
            elif item.quantity > 0:
                new_inventory.append(item)

        self.inventory = new_inventory
        reward_state.overhead_penalty += total_items * 0.01

        # Advance step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if not done:
            self._update_environment_dynamics()

        # Score: (net_profit / max_potential) - waste_ratio.
        # max_potential is frozen at reset, so profit_ratio is a stable signal throughout.
        # waste_ratio denominator includes waste_value to stay bounded in (0, 1).
        waste_ratio = min(1.0, self.total_waste_value / max(1.0, self.total_waste_value + sum(
            item.quantity * item.cost_price for item in self.inventory
        ) + abs(self.total_net_profit)))

        raw_profit_ratio = max(0.0, self.total_net_profit / self.max_potential_profit)
        profit_ratio_clamped = min(1.0, raw_profit_ratio)
        score = max(0.0, profit_ratio_clamped - waste_ratio)

        if verbose:
            print(f"  [Score] Net Profit=${self.total_net_profit:.2f} | Max Potential=${self.max_potential_profit:.2f} | "
                  f"Profit Ratio={profit_ratio_clamped:.4f} | Waste Ratio={waste_ratio:.4f} -> Score={score:.4f}")

        return StepResult(
            observation=self.state(),
            reward=reward_state.total_reward,
            reward_breakdown=reward_state,
            done=done,
            score=score,
            info={
                "net_profit": self.total_net_profit,
                "waste_value": self.total_waste_value,
                "task": self.task_name,
            },
        )

    def _sell_inventory(self, p_id: str, expected_sales: int, price: float, reward_state: RewardState) -> int:
        """
        Sell from inventory using FIFO (oldest batches first — highest expiry risk).
        Returns actual units sold.

        Selling below cost is no longer double-penalized as 'waste' — the agent
        loses margin (negative profit_margin) but is not hit with an additional
        waste_penalty. This allows rational fire-sale behavior for perishables.
        """
        remaining_qty = expected_sales
        sales_made = 0

        p_items = [item for item in self.inventory if item.product_id == p_id]
        p_items.sort(key=lambda x: x.time_to_expiry_steps)  # FIFO: sell soonest-to-expire first

        for item in p_items:
            if remaining_qty <= 0:
                break

            sell_qty = min(item.quantity, remaining_qty)
            item.quantity -= sell_qty
            remaining_qty -= sell_qty
            sales_made += sell_qty

            profit_margin = (price - item.cost_price) * sell_qty
            self.total_net_profit += profit_margin

            if profit_margin >= 0:
                reward_state.successful_sale_reward += profit_margin
            # If profit_margin < 0 (fire sale), no additional waste_penalty — the agent
            # already accepts a lower total_net_profit, which reduces the score naturally.

            # Bonus for clearing near-expiry stock profitably
            if item.time_to_expiry_steps <= 4 and profit_margin >= 0:
                reward_state.efficiency_bonus += sell_qty * 0.5

        return sales_made

    def process_manual_sale(self, p_id: str, quantity: int, price: float) -> int:
        """
        Human-in-the-loop bridge.
        Instantly deducts inventory and attributes the manual GUI sale profit to the AI's episodic score.
        Returns the actual number of units fulfilled based on available stock.
        """
        # Create a dummy reward state to capture the profit
        dummy_state = RewardState()
        
        # We reuse the FIFO _sell_inventory logic to properly handle rotting batches.
        sales_achieved = self._sell_inventory(p_id, quantity, price, dummy_state)
        
        # Note: self.total_net_profit is already bumped inside _sell_inventory!
        # Because we used a dummy state, we don't need to add successful_sale_reward into the current tick 
        # (the tick computes later), but the absolute total_net_profit math tracks it for the final score.
        print(f"  [MANUAL OVERRIDE] Human Buyer explicitly ordered {sales_achieved}x {p_id} at ${price:.2f}!")
        return sales_achieved

    def _discard_inventory(self, p_id: str, quantity: int, reward_state: RewardState, manual: bool):
        """Discard inventory FIFO (oldest first). Manual discard costs 1.0x, expiry costs 1.5x."""
        remaining_qty = quantity
        p_items = [item for item in self.inventory if item.product_id == p_id]
        p_items.sort(key=lambda x: x.time_to_expiry_steps)

        for item in p_items:
            if remaining_qty <= 0:
                break
            discard_qty = min(item.quantity, remaining_qty)
            item.quantity -= discard_qty
            remaining_qty -= discard_qty

            loss = discard_qty * item.cost_price
            self.total_waste_value += loss
            reward_state.waste_penalty += loss * (1.0 if manual else 1.5)
