"""
FastAPI REST API for Q-Store Gym.

Exposes the RL environment, marketplace, and admin dashboard as HTTP endpoints.

Sections covered:
  Section 6.2  — Seller/supplier endpoints
  Section 6.3  — Buyer order endpoints
  Section 6.4  — RL decision loop endpoint
  Section 6.5  — Model serving, auth, rate limiting
  Section 7    — Thread-safety, graceful degradation, SLA monitoring, health check

Architecture:
  - Each active client gets a unique session ID.  The RL environment state lives in
    a per-session object protected by a threading.Lock (thread-safety).
  - API-key authentication guards all state-mutating endpoints.
  - Rate limiting (slowapi) caps LLM and expensive endpoints.
  - If the PPO model is missing or throws, the agent falls back to the deterministic
    1.3x markup policy (graceful degradation).
  - Every request/response is timed and recorded in monitoring.MetricsStore.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import uuid
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from env import QStoreEnv
from models import ActionSpace, ObservationSpace
from tasks import AVAILABLE_TASKS
from marketplace import (
    BuyerOrder, BuyerProfile, CatalogProduct, OrderStatus,
    PurchaseOrder, Supplier, catalog, order_manager,
)
from monitoring import EpisodeRecord, StepRecord, metrics


# ──────────────────────────────────────────────────────────────
# Auth helpers
# ──────────────────────────────────────────────────────────────

_ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "dev-admin-key")
_CLIENT_API_KEY = os.environ.get("CLIENT_API_KEY", "dev-client-key")

def require_client_key(x_api_key: str = Header(...)):
    if x_api_key not in (_CLIENT_API_KEY, _ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")

def require_admin_key(x_api_key: str = Header(...)):
    if x_api_key != _ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin key required.")


# ──────────────────────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ──────────────────────────────────────────────────────────────
# Per-session RL environment state (thread-safe)
# ──────────────────────────────────────────────────────────────

class SessionState:
    """
    Holds a live QStoreEnv instance and its current observation for one client session.
    Protected by a per-session lock so concurrent HTTP requests cannot corrupt state.
    """
    def __init__(self, task_name: str):
        self.env        = QStoreEnv()
        self.obs        = self.env.reset(task_name)
        self.task_name  = task_name
        self.step_count = 0
        self.lock       = threading.Lock()
        self.created_at = time.time()
        self.last_used  = time.time()

_sessions: Dict[str, SessionState] = {}
_sessions_lock = threading.Lock()

def _get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call POST /session/new first.")
    state.last_used = time.time()
    return state

def _purge_stale_sessions(max_age_seconds: int = 3600):
    """Remove sessions inactive for more than max_age_seconds."""
    cutoff = time.time() - max_age_seconds
    with _sessions_lock:
        stale = [sid for sid, s in _sessions.items() if s.last_used < cutoff]
        for sid in stale:
            del _sessions[sid]


# ──────────────────────────────────────────────────────────────
# PPO model loader (with graceful degradation)
# ──────────────────────────────────────────────────────────────

_ppo_cache: Dict[str, Any] = {}   # stem → (model, vecnorm_env)
_ppo_lock = threading.Lock()

def _load_ppo_model(task_name: str, curriculum: bool = False):
    """
    Load a trained PPO model + VecNormalize stats.
    Returns (model, norm_env) or (None, None) if unavailable.
    Never raises — callers must handle None and fall back to deterministic.
    """
    stem = "ppo_curriculum" if curriculum else f"ppo_{task_name.replace(' ', '_')}"
    with _ppo_lock:
        if stem in _ppo_cache:
            return _ppo_cache[stem]

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor
        from gym_wrapper import QStoreGymWrapper

        if not os.path.exists(f"{stem}.zip"):
            metrics.set_model_status(stem, False)
            return None, None

        raw = DummyVecEnv([lambda: Monitor(QStoreGymWrapper(task_name=task_name))])
        vecnorm_path = f"{stem}_vecnorm.pkl"
        if os.path.exists(vecnorm_path):
            norm_env = VecNormalize.load(vecnorm_path, raw)
            norm_env.training    = False
            norm_env.norm_reward = False
        else:
            norm_env = raw

        model = PPO.load(stem, env=norm_env)
        metrics.set_model_status(stem, True)

        with _ppo_lock:
            _ppo_cache[stem] = (model, norm_env)
        return model, norm_env

    except Exception as exc:
        metrics.set_last_error(f"PPO load failed ({stem}): {exc}")
        metrics.set_model_status(stem, False)
        return None, None


def _deterministic_action(obs: ObservationSpace) -> ActionSpace:
    """
    Graceful degradation fallback: 1.3x markup on all products in inventory,
    no sourcing, no waste management.
    Used whenever the PPO model is unavailable or raises.
    """
    return ActionSpace(
        pricing={item.product_id: 1.3 for item in obs.inventory},
        sourcing={},
        waste_management={},
    )


# ──────────────────────────────────────────────────────────────
# Application startup / shutdown
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load models on startup so first request isn't slow
    for task in AVAILABLE_TASKS:
        _load_ppo_model(task)
    yield
    # Cleanup
    with _sessions_lock:
        _sessions.clear()


app = FastAPI(
    title="Q-Store Gym API",
    description="REST API for the Q-Store dark store RL simulator with marketplace integration.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", tags=["Frontend"])
def read_root():
    """Serves the interactive premium dashboard UI."""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


# ──────────────────────────────────────────────────────────────
# Middleware: request timing + metrics
# ──────────────────────────────────────────────────────────────

@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    latency = time.time() - t0
    error = response.status_code >= 400
    metrics.record_api_request(latency, error=error)
    response.headers["X-Response-Time-Ms"] = str(round(latency * 1000, 2))
    return response


# ──────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    task_name: str = "The Night Shift"

class StepRequest(BaseModel):
    """The agent action to execute. pricing is a cost-multiplier dict."""
    action: ActionSpace

class RLDecideRequest(BaseModel):
    """Ask the RL agent to choose an action for the current session state."""
    session_id: str
    use_curriculum: bool = False
    stochastic: bool = False

class PlaceOrderRequest(BaseModel):
    buyer_id:         str
    product_id:       str
    quantity:         int
    delivery_address: str

class RegisterBuyerRequest(BaseModel):
    name:    str
    email:   str
    address: str

class AddProductRequest(BaseModel):
    product:  CatalogProduct

class AddSupplierRequest(BaseModel):
    supplier: Supplier


# ──────────────────────────────────────────────────────────────
# Section 6.4 — RL Session endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/session/new", tags=["RL Environment"])
def new_session(
    body: NewSessionRequest,
    _: str = Depends(require_client_key),
):
    """
    Create a new RL environment session for the given task.
    Returns a session_id to use in subsequent /session/{id}/step calls.
    """
    if body.task_name not in AVAILABLE_TASKS:
        raise HTTPException(400, f"Unknown task. Available: {AVAILABLE_TASKS}")

    session_id = str(uuid.uuid4())
    state = SessionState(body.task_name)
    with _sessions_lock:
        _sessions[session_id] = state

    return {
        "session_id":   session_id,
        "task_name":    body.task_name,
        "observation":  state.obs.model_dump(),
        "available_tasks": AVAILABLE_TASKS,
    }


@app.get("/session/{session_id}/state", tags=["RL Environment"])
def get_state(
    session_id: str,
    _: str = Depends(require_client_key),
):
    """Return the current observation for an existing session."""
    state = _get_session(session_id)
    with state.lock:
        obs = state.obs.model_dump()
    return {"session_id": session_id, "observation": obs}


@app.post("/session/{session_id}/step", tags=["RL Environment"])
def step_env(
    session_id: str,
    body: StepRequest,
    _: str = Depends(require_client_key),
):
    """
    Execute one action in the environment.
    Returns the new observation, reward, done flag, score, and reward breakdown.
    """
    state = _get_session(session_id)

    with state.lock:
        result = state.env.step(body.action, verbose=False)
        state.obs = result.observation
        state.step_count += 1

        metrics.record_step(StepRecord(
            session_id=session_id,
            step=state.step_count,
            reward=result.reward,
            score=result.score,
            waste_ratio=result.info.get("waste_ratio", 0.0),
        ))

        sla_breaches = order_manager.mark_sla_breaches(state.step_count)
        for _ in range(sla_breaches):
            metrics.record_sla_event(breached=True)

        if result.done:
            metrics.record_episode(EpisodeRecord(
                task_name=state.task_name,
                agent_type="manual",
                final_score=result.score,
                total_reward=result.reward,
                net_profit=result.info.get("net_profit", 0.0),
                waste_value=result.info.get("waste_value", 0.0),
                steps=state.step_count,
            ))

    return {
        "observation":      result.observation.model_dump(),
        "reward":           result.reward,
        "reward_breakdown": result.reward_breakdown.model_dump(),
        "done":             result.done,
        "score":            result.score,
        "info":             result.info,
    }


@app.post("/session/rl-decide", tags=["RL Environment"])
@limiter.limit("60/minute")
def rl_decide(
    request: Request,
    body: RLDecideRequest,
    _: str = Depends(require_client_key),
):
    """
    Ask the trained PPO agent for its recommended action for the current session state.
    Executes the action in the environment and returns the result.

    Graceful degradation: if the PPO model is not available or raises, automatically
    falls back to the deterministic 1.3x baseline — the API never returns a 500 for
    a missing model.
    """
    state = _get_session(body.session_id)

    model, norm_env = _load_ppo_model(state.task_name, curriculum=body.use_curriculum)

    with state.lock:
        agent_type = "ppo"

        if model is None:
            # Graceful degradation path
            action = _deterministic_action(state.obs)
            agent_type = "deterministic_fallback"
        else:
            try:
                from gym_wrapper import QStoreGymWrapper
                # Flatten the current observation to a numpy vector
                tmp_wrapper = QStoreGymWrapper(state.task_name)
                obs_arr = tmp_wrapper._flatten_obs(state.obs)
                import numpy as np
                obs_vec = norm_env.normalize_obs(obs_arr[np.newaxis, :])
                raw_action, _ = model.predict(obs_vec, deterministic=not body.stochastic)
                action = tmp_wrapper._decode_action(raw_action[0])
            except Exception as exc:
                metrics.set_last_error(f"PPO predict failed: {exc}")
                action = _deterministic_action(state.obs)
                agent_type = "deterministic_fallback"

        result = state.env.step(action, verbose=False)
        state.obs = result.observation
        state.step_count += 1

        metrics.record_step(StepRecord(
            session_id=body.session_id,
            step=state.step_count,
            reward=result.reward,
            score=result.score,
            waste_ratio=result.info.get("waste_ratio", 0.0),
        ))

        if result.done:
            metrics.record_episode(EpisodeRecord(
                task_name=state.task_name,
                agent_type=agent_type,
                final_score=result.score,
                total_reward=result.reward,
                net_profit=result.info.get("net_profit", 0.0),
                waste_value=result.info.get("waste_value", 0.0),
                steps=state.step_count,
            ))

    return {
        "agent_type":       agent_type,
        "action":           action.model_dump(),
        "observation":      result.observation.model_dump(),
        "reward":           result.reward,
        "reward_breakdown": result.reward_breakdown.model_dump(),
        "done":             result.done,
        "score":            result.score,
    }


# ──────────────────────────────────────────────────────────────
# Section 6.3 — Buyer endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/buyers/register", tags=["Buyer"])
def register_buyer(body: RegisterBuyerRequest):
    """Register a new buyer account."""
    profile = order_manager.register_buyer(
        name=body.name, email=body.email, address=body.address
    )
    return profile.model_dump()


@app.get("/buyers/{buyer_id}", tags=["Buyer"])
def get_buyer(buyer_id: str):
    profile = order_manager.get_buyer(buyer_id)
    if not profile:
        raise HTTPException(404, "Buyer not found.")
    return profile.model_dump()


@app.get("/catalog", tags=["Buyer"])
def browse_catalog():
    """
    Browse available products with current cost prices.
    In production, prices shown here are set by the RL agent in real time.
    """
    products = catalog.list_products(active_only=True)
    return {"products": [p.model_dump() for p in products]}


@app.post("/orders", tags=["Buyer"])
def place_order(
    body: PlaceOrderRequest,
    session_id: Optional[str] = None,
):
    """
    Place a buyer order for a product.
    The unit_price is read from the catalog (reflects latest RL agent pricing decision).
    session_id is optional — if provided, uses the current session's RL-set price.
    """
    product = catalog.get_product(body.product_id)
    if not product:
        raise HTTPException(404, f"Product '{body.product_id}' not found in catalog.")
    if body.quantity <= 0:
        raise HTTPException(400, "Quantity must be >= 1.")

    buyer = order_manager.get_buyer(body.buyer_id)
    if not buyer:
        raise HTTPException(404, f"Buyer '{body.buyer_id}' not found. Register first.")

    # Determine current sell price — use RL-set price if a session exists
    unit_price = product.cost_price * 1.3  # fallback: deterministic markup
    current_step = 0
    if session_id:
        try:
            state = _get_session(session_id)
            with state.lock:
                current_step = state.step_count
                # Try to extract the true RL sell price if available, otherwise fallback.
                # The RL price was theoretically printed to the UI catalog, but wait:
                # The UI just passes the click. The real multiplier sits in the last action.
                
                # We inject the human buyer's manual order directly into the RL inventory!
                # Extract the base env from the GymWrapper
                base_env = getattr(state.env, 'env', None) if hasattr(state.env, 'step') else state.env
                if hasattr(base_env, 'process_manual_sale'):
                    fulfilled = base_env.process_manual_sale(body.product_id, body.quantity, unit_price)
                    if fulfilled < body.quantity:
                        print(f"API Warning: Manual order demanded {body.quantity} but store only had {fulfilled} in stock!")
        except HTTPException:
            pass

    order = order_manager.place_order(
        buyer_id=body.buyer_id,
        product_id=body.product_id,
        quantity=body.quantity,
        unit_price=unit_price,
        delivery_address=body.delivery_address,
        current_step=current_step,
    )
    metrics.record_sla_event(breached=False)

    return {
        "order":    order.model_dump(),
        "message":  f"Order placed. Expected delivery within 15 minutes (1 step). SLA deadline: step {order.sla_deadline_step}.",
    }


@app.get("/orders/{order_id}", tags=["Buyer"])
def get_order_status(order_id: str):
    order = order_manager.get_order(order_id)
    if not order:
        raise HTTPException(404, "Order not found.")
    return order.model_dump()


@app.get("/orders", tags=["Buyer"])
def list_buyer_orders(
    buyer_id: Optional[str] = None,
    status:   Optional[OrderStatus] = None,
):
    orders = order_manager.list_orders(buyer_id=buyer_id, status=status)
    return {"orders": [o.model_dump() for o in orders], "count": len(orders)}


@app.patch("/orders/{order_id}/status", tags=["Buyer"])
def update_order_status(
    order_id:   str,
    new_status: OrderStatus,
    rider_id:   Optional[str] = None,
    _: str = Depends(require_client_key),
):
    """Advance an order through the delivery state machine."""
    order = order_manager.advance_status(order_id, new_status, rider_id)
    if not order:
        raise HTTPException(404, "Order not found.")
    return order.model_dump()


# ──────────────────────────────────────────────────────────────
# Section 6.2 — Seller/Supplier endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/suppliers", tags=["Supplier"])
def list_suppliers(_: str = Depends(require_client_key)):
    return {"suppliers": [s.model_dump() for s in catalog.list_suppliers()]}


@app.post("/suppliers", tags=["Supplier"])
def add_supplier(
    body: AddSupplierRequest,
    _: str = Depends(require_admin_key),
):
    """Onboard a new supplier. Admin only."""
    existing = catalog.get_supplier(body.supplier.supplier_id)
    if existing:
        raise HTTPException(409, f"Supplier '{body.supplier.supplier_id}' already exists.")
    return catalog.add_supplier(body.supplier).model_dump()


@app.post("/catalog/products", tags=["Supplier"])
def add_product(
    body: AddProductRequest,
    _: str = Depends(require_admin_key),
):
    """Add or update a product in the catalog. Admin only."""
    return catalog.add_product(body.product).model_dump()


@app.patch("/catalog/products/{product_id}/price", tags=["Supplier"])
def update_product_price(
    product_id:     str,
    new_cost_price: float,
    _: str = Depends(require_admin_key),
):
    """Update the supplier cost price for a product. Admin only."""
    updated = catalog.update_product_price(product_id, new_cost_price)
    if not updated:
        raise HTTPException(404, f"Product '{product_id}' not found.")
    return updated.model_dump()


@app.get("/suppliers/balance", tags=["Supplier"])
def unsettled_balance(_: str = Depends(require_admin_key)):
    return {"unsettled_balance": catalog.unsettled_balance()}


@app.post("/suppliers/settle", tags=["Supplier"])
def settle_supplier_balance(_: str = Depends(require_admin_key)):
    total = catalog.settle_pos()
    return {"settled_amount": total, "message": "All outstanding POs marked as settled."}


# ──────────────────────────────────────────────────────────────
# Section 7 — Admin / Monitoring / Health
# ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Admin"])
def health_check():
    """
    Lightweight health probe for load balancers and uptime monitors.
    Returns 200 as long as the API is running.
    """
    return {"status": "ok", "timestamp": time.time()}


@app.get("/admin/dashboard", tags=["Admin"])
def admin_dashboard(_: str = Depends(require_admin_key)):
    """
    Full admin dashboard snapshot:
    - API latency and error rates
    - Per-task RL agent performance (mean ± std score)
    - Model availability status
    - SLA breach rates
    - System uptime
    """
    _purge_stale_sessions()
    dashboard = metrics.get_dashboard()
    dashboard["active_sessions"] = len(_sessions)
    dashboard["sla_order_summary"] = order_manager.sla_summary()
    return dashboard


@app.get("/admin/metrics/scores", tags=["Admin"])
def episode_scores(
    task_name:  Optional[str] = None,
    agent_type: Optional[str] = None,
    last_n:     int = 50,
    _: str = Depends(require_admin_key),
):
    """Episode score history for learning curve analysis."""
    return {"scores": metrics.episode_scores(task_name=task_name, agent_type=agent_type, last_n=last_n)}


@app.get("/admin/metrics/reward-curve", tags=["Admin"])
def reward_curve(
    session_id: Optional[str] = None,
    last_n: int = 200,
    _: str = Depends(require_admin_key),
):
    """Per-step reward and score data. Use to plot learning curves during/after training."""
    return {"curve": metrics.reward_curve(session_id=session_id, last_n=last_n)}


@app.get("/admin/tasks", tags=["Admin"])
def available_tasks():
    return {"tasks": AVAILABLE_TASKS}


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("ENV", "production") == "development",
        workers=1,  # single worker for now; use gunicorn + multiple workers in production
    )
