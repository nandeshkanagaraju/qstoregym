"""
Marketplace layer for Q-Store Gym.

Provides the data models and in-memory state management for:
  - Supplier catalog (Section 6.2)
  - Product catalog (Section 6.2)
  - Buyer order flow (Section 6.3)
  - Order state machine: pending → processing → picked → out_for_delivery → delivered
  - Settlement tracking (what the store owes each supplier)

In production, replace the in-memory stores with PostgreSQL (SQLAlchemy models are
stubbed at the bottom of this file). Redis should handle real-time order state.
"""

import uuid
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────

class OrderStatus(str, Enum):
    PENDING          = "pending"
    PROCESSING       = "processing"
    PICKED           = "picked"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED        = "delivered"
    CANCELLED        = "cancelled"


class SupplierReliability(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ──────────────────────────────────────────────────────────────
# Supplier & Product catalog models
# ──────────────────────────────────────────────────────────────

class Supplier(BaseModel):
    """A registered supplier who can fulfill sourcing orders from the dark store."""
    supplier_id:       str
    name:              str
    reliability:       SupplierReliability = SupplierReliability.MEDIUM
    avg_lead_time_steps: int = 4            # steps until delivery (1 step = 15 min)
    contact_email:     str = ""
    active:            bool = True


class CatalogProduct(BaseModel):
    """
    A product entry in the supplier catalog.
    Replaces the hardcoded cost_price values in tasks.py for production use.
    The RL agent's cost_price is read from here at environment reset.
    """
    product_id:         str
    name:               str
    category:           str
    unit:               str                 # "litre", "loaf", "bag", "punnet"
    cost_price:         float               # what the store pays the supplier
    shelf_life_steps:   int                 # steps until expiry (task config uses this)
    min_order_qty:      int = 1
    supplier_id:        str
    active:             bool = True


class PurchaseOrder(BaseModel):
    """A sourcing order placed by the RL agent to a supplier."""
    po_id:          str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    product_id:     str
    supplier_id:    str
    quantity:       int
    unit_cost:      float
    total_cost:     float
    placed_at:      datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expected_arrival_step: int = 0
    settled:        bool = False            # whether the store has paid


# ──────────────────────────────────────────────────────────────
# Buyer order models
# ──────────────────────────────────────────────────────────────

class BuyerOrder(BaseModel):
    """A customer order placed through the buyer-facing interface."""
    order_id:           str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    buyer_id:           str
    product_id:         str
    quantity:           int
    unit_price:         float               # price set by RL agent at time of order
    total_price:        float
    delivery_address:   str
    status:             OrderStatus = OrderStatus.PENDING
    placed_at:          datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_delivery: Optional[datetime] = None
    delivered_at:       Optional[datetime] = None
    rider_id:           Optional[str] = None
    # SLA tracking — delivery must complete within 10 min (≈ 1 step) for Q-Commerce promise
    sla_deadline_step:  int = 0
    sla_breached:       bool = False


class BuyerProfile(BaseModel):
    """Registered buyer account."""
    buyer_id:       str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:           str
    email:          str
    address:        str
    created_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_orders:   int = 0
    total_spent:    float = 0.0


# ──────────────────────────────────────────────────────────────
# Catalog manager (in-memory — swap to PostgreSQL in production)
# ──────────────────────────────────────────────────────────────

class SupplierCatalog:
    """
    Thread-safe in-memory product and supplier catalog.
    Pre-seeded with the Q-Store Gym default products so the RL environment
    can read cost_price and shelf_life_steps from a live data source.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._suppliers:  Dict[str, Supplier]        = {}
        self._products:   Dict[str, CatalogProduct]  = {}
        self._pos:        List[PurchaseOrder]         = []
        self._seed_defaults()

    def _seed_defaults(self):
        """Pre-seed catalog with the four default Q-Store products."""
        fresh_supplier = Supplier(
            supplier_id="sup_fresh",
            name="FreshCo Wholesale",
            reliability=SupplierReliability.HIGH,
            avg_lead_time_steps=4,
            contact_email="orders@freshco.example",
        )
        dry_supplier = Supplier(
            supplier_id="sup_dry",
            name="DryGoods Direct",
            reliability=SupplierReliability.MEDIUM,
            avg_lead_time_steps=4,
            contact_email="orders@drygoodsdirect.example",
        )
        self._suppliers[fresh_supplier.supplier_id] = fresh_supplier
        self._suppliers[dry_supplier.supplier_id]   = dry_supplier

        defaults = [
            CatalogProduct(product_id="milk",         name="Full-Fat Milk (1L)",
                           category="dairy",    unit="litre",   cost_price=2.0,
                           shelf_life_steps=48,  min_order_qty=5,  supplier_id="sup_fresh"),
            CatalogProduct(product_id="bread",        name="White Sliced Bread",
                           category="bakery",   unit="loaf",    cost_price=1.5,
                           shelf_life_steps=24,  min_order_qty=5,  supplier_id="sup_fresh"),
            CatalogProduct(product_id="chips",        name="Ready Salted Crisps (150g)",
                           category="snacks",   unit="bag",     cost_price=1.0,
                           shelf_life_steps=100, min_order_qty=10, supplier_id="sup_dry"),
            CatalogProduct(product_id="strawberries", name="Strawberries (400g punnet)",
                           category="produce",  unit="punnet",  cost_price=4.0,
                           shelf_life_steps=10,  min_order_qty=5,  supplier_id="sup_fresh"),
        ]
        for p in defaults:
            self._products[p.product_id] = p

    # ── Read operations ──────────────────────────────────────

    def get_product(self, product_id: str) -> Optional[CatalogProduct]:
        with self._lock:
            return self._products.get(product_id)

    def list_products(self, active_only: bool = True) -> List[CatalogProduct]:
        with self._lock:
            products = list(self._products.values())
        return [p for p in products if p.active] if active_only else products

    def get_supplier(self, supplier_id: str) -> Optional[Supplier]:
        with self._lock:
            return self._suppliers.get(supplier_id)

    def list_suppliers(self, active_only: bool = True) -> List[Supplier]:
        with self._lock:
            suppliers = list(self._suppliers.values())
        return [s for s in suppliers if s.active] if active_only else suppliers

    # ── Write operations ─────────────────────────────────────

    def add_product(self, product: CatalogProduct) -> CatalogProduct:
        with self._lock:
            self._products[product.product_id] = product
        return product

    def update_product_price(self, product_id: str, new_cost_price: float) -> Optional[CatalogProduct]:
        with self._lock:
            p = self._products.get(product_id)
            if p is None:
                return None
            updated = p.model_copy(update={"cost_price": new_cost_price})
            self._products[product_id] = updated
        return updated

    def add_supplier(self, supplier: Supplier) -> Supplier:
        with self._lock:
            self._suppliers[supplier.supplier_id] = supplier
        return supplier

    def record_purchase_order(self, po: PurchaseOrder) -> PurchaseOrder:
        with self._lock:
            self._pos.append(po)
        return po

    def unsettled_balance(self) -> float:
        """Total amount owed to suppliers for unsettled POs."""
        with self._lock:
            return sum(po.total_cost for po in self._pos if not po.settled)

    def settle_pos(self) -> float:
        """Mark all outstanding POs as settled. Returns total amount settled."""
        with self._lock:
            total = sum(po.total_cost for po in self._pos if not po.settled)
            for po in self._pos:
                po.settled = True
        return total


# ──────────────────────────────────────────────────────────────
# Order manager (in-memory — swap to PostgreSQL + Redis in production)
# ──────────────────────────────────────────────────────────────

class OrderManager:
    """
    Thread-safe in-memory order book for buyer orders.
    Tracks the full order lifecycle from placement to delivery.
    Each buyer order contributes to the real-world demand signal that
    replaces the simulated demand_index in production integration.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._orders: Dict[str, BuyerOrder]   = {}
        self._buyers: Dict[str, BuyerProfile] = {}

    # ── Buyer registration ───────────────────────────────────

    def register_buyer(self, name: str, email: str, address: str) -> BuyerProfile:
        profile = BuyerProfile(name=name, email=email, address=address)
        with self._lock:
            self._buyers[profile.buyer_id] = profile
        return profile

    def get_buyer(self, buyer_id: str) -> Optional[BuyerProfile]:
        with self._lock:
            return self._buyers.get(buyer_id)

    # ── Order placement ──────────────────────────────────────

    def place_order(
        self,
        buyer_id:         str,
        product_id:       str,
        quantity:         int,
        unit_price:       float,
        delivery_address: str,
        current_step:     int,
    ) -> BuyerOrder:
        order = BuyerOrder(
            buyer_id=buyer_id,
            product_id=product_id,
            quantity=quantity,
            unit_price=unit_price,
            total_price=round(unit_price * quantity, 2),
            delivery_address=delivery_address,
            sla_deadline_step=current_step + 1,  # Q-Commerce SLA: deliver within 1 step (15 min)
        )
        with self._lock:
            self._orders[order.order_id] = order
            buyer = self._buyers.get(buyer_id)
            if buyer:
                buyer.total_orders += 1
                buyer.total_spent  += order.total_price
        return order

    # ── Order status transitions ─────────────────────────────

    def advance_status(self, order_id: str, new_status: OrderStatus,
                       rider_id: Optional[str] = None) -> Optional[BuyerOrder]:
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return None
            order.status = new_status
            if rider_id:
                order.rider_id = rider_id
            if new_status == OrderStatus.DELIVERED:
                order.delivered_at = datetime.now(timezone.utc)
        return order

    def mark_sla_breaches(self, current_step: int) -> int:
        """Check all pending orders against SLA deadline. Returns count of newly breached orders."""
        breached = 0
        with self._lock:
            for order in self._orders.values():
                if (not order.sla_breached
                        and order.status not in (OrderStatus.DELIVERED, OrderStatus.CANCELLED)
                        and current_step > order.sla_deadline_step):
                    order.sla_breached = True
                    breached += 1
        return breached

    # ── Read operations ──────────────────────────────────────

    def get_order(self, order_id: str) -> Optional[BuyerOrder]:
        with self._lock:
            return self._orders.get(order_id)

    def list_orders(
        self,
        buyer_id: Optional[str] = None,
        status:   Optional[OrderStatus] = None,
    ) -> List[BuyerOrder]:
        with self._lock:
            orders = list(self._orders.values())
        if buyer_id:
            orders = [o for o in orders if o.buyer_id == buyer_id]
        if status:
            orders = [o for o in orders if o.status == status]
        return orders

    def pending_demand(self) -> Dict[str, int]:
        """
        Returns {product_id: total_quantity} for all pending/processing orders.
        This is the real-world demand signal that replaces simulated demand_index
        when the marketplace is integrated with the live RL environment.
        """
        demand: Dict[str, int] = {}
        with self._lock:
            for order in self._orders.values():
                if order.status in (OrderStatus.PENDING, OrderStatus.PROCESSING):
                    demand[order.product_id] = demand.get(order.product_id, 0) + order.quantity
        return demand

    def sla_summary(self) -> Dict[str, int]:
        with self._lock:
            total    = len(self._orders)
            breached = sum(1 for o in self._orders.values() if o.sla_breached)
            delivered = sum(1 for o in self._orders.values() if o.status == OrderStatus.DELIVERED)
        return {
            "total_orders":    total,
            "delivered":       delivered,
            "sla_breached":    breached,
            "sla_breach_rate": round(breached / max(1, total), 4),
        }


# ──────────────────────────────────────────────────────────────
# Module-level singletons (shared across API workers via import)
# ──────────────────────────────────────────────────────────────

catalog       = SupplierCatalog()
order_manager = OrderManager()
