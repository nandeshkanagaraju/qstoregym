"""
Monitoring, metrics, and SLA tracking for Q-Store Gym.

Tracks:
  - Episode scores and reward curves (so you can see if learning is happening)
  - Per-step RL agent decisions
  - Waste ratio trends
  - SLA delivery compliance
  - System health (model availability, API latency)

All state is in-memory. In production, push metrics to Prometheus/Grafana or
a time-series database. The `get_dashboard()` method returns the full snapshot
that the admin API endpoint exposes.
"""

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────

class EpisodeRecord:
    """Metrics captured at the end of a single training/inference episode."""
    __slots__ = (
        "task_name", "agent_type", "final_score", "total_reward",
        "net_profit", "waste_value", "steps", "timestamp",
    )

    def __init__(
        self,
        task_name:    str,
        agent_type:   str,      # "ppo", "gpt", "deterministic"
        final_score:  float,
        total_reward: float,
        net_profit:   float,
        waste_value:  float,
        steps:        int,
    ):
        self.task_name    = task_name
        self.agent_type   = agent_type
        self.final_score  = final_score
        self.total_reward = total_reward
        self.net_profit   = net_profit
        self.waste_value  = waste_value
        self.steps        = steps
        self.timestamp    = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


class StepRecord:
    """Metrics captured at each environment step — used for reward curve and action logging."""
    __slots__ = ("session_id", "step", "reward", "score", "waste_ratio", "timestamp")

    def __init__(self, session_id: str, step: int, reward: float, score: float, waste_ratio: float):
        self.session_id  = session_id
        self.step        = step
        self.reward      = reward
        self.score       = score
        self.waste_ratio = waste_ratio
        self.timestamp   = time.time()


# ──────────────────────────────────────────────────────────────
# Metrics store
# ──────────────────────────────────────────────────────────────

class MetricsStore:
    """
    Thread-safe in-memory metrics collection.

    Keeps:
    - Last 1000 episode records across all tasks/agents
    - Last 10,000 step records (reward curve source)
    - Running counters for API requests and latencies
    - System health flags (model loaded, last error)
    """

    def __init__(self, max_episodes: int = 1000, max_steps: int = 10_000):
        self._lock = threading.Lock()

        self._episodes: Deque[EpisodeRecord] = deque(maxlen=max_episodes)
        self._steps:    Deque[StepRecord]    = deque(maxlen=max_steps)

        # API request counters
        self._api_requests:   int   = 0
        self._api_errors:     int   = 0
        self._total_latency:  float = 0.0  # seconds

        # Model health
        self._models_loaded:  Dict[str, bool]  = {}
        self._last_error:     Optional[str]    = None
        self._startup_time:   float            = time.time()

        # SLA counters
        self._sla_total:      int = 0
        self._sla_breached:   int = 0

    # ── Recording ────────────────────────────────────────────

    def record_episode(self, record: EpisodeRecord):
        with self._lock:
            self._episodes.append(record)

    def record_step(self, record: StepRecord):
        with self._lock:
            self._steps.append(record)

    def record_api_request(self, latency_seconds: float, error: bool = False):
        with self._lock:
            self._api_requests  += 1
            self._total_latency += latency_seconds
            if error:
                self._api_errors += 1

    def record_sla_event(self, breached: bool):
        with self._lock:
            self._sla_total   += 1
            if breached:
                self._sla_breached += 1

    def set_model_status(self, model_name: str, loaded: bool):
        with self._lock:
            self._models_loaded[model_name] = loaded

    def set_last_error(self, error: str):
        with self._lock:
            self._last_error = error

    # ── Queries ──────────────────────────────────────────────

    def episode_scores(
        self,
        task_name:  Optional[str] = None,
        agent_type: Optional[str] = None,
        last_n:     int = 100,
    ) -> List[Dict[str, Any]]:
        """Return recent episode records, optionally filtered."""
        with self._lock:
            records = list(self._episodes)
        if task_name:
            records = [r for r in records if r.task_name == task_name]
        if agent_type:
            records = [r for r in records if r.agent_type == agent_type]
        return [r.to_dict() for r in records[-last_n:]]

    def reward_curve(self, session_id: Optional[str] = None, last_n: int = 500) -> List[Dict]:
        """Return per-step reward and score data for plotting learning curves."""
        with self._lock:
            steps = list(self._steps)
        if session_id:
            steps = [s for s in steps if s.session_id == session_id]
        return [
            {"step": s.step, "reward": s.reward, "score": s.score,
             "waste_ratio": s.waste_ratio, "t": s.timestamp}
            for s in steps[-last_n:]
        ]

    def per_task_summary(self) -> Dict[str, Dict]:
        """Mean ± std score for each task, by agent type."""
        with self._lock:
            records = list(self._episodes)

        summary: Dict[str, Dict] = {}
        for r in records:
            key = f"{r.task_name}::{r.agent_type}"
            if key not in summary:
                summary[key] = {"task": r.task_name, "agent": r.agent_type, "scores": []}
            summary[key]["scores"].append(r.final_score)

        result = {}
        for key, data in summary.items():
            scores = data["scores"]
            n = len(scores)
            mean = sum(scores) / n if n else 0.0
            variance = sum((s - mean) ** 2 for s in scores) / max(1, n - 1)
            std = variance ** 0.5
            result[key] = {
                "task":   data["task"],
                "agent":  data["agent"],
                "n":      n,
                "mean":   round(mean, 4),
                "std":    round(std, 4),
                "best":   round(max(scores), 4) if scores else 0.0,
                "worst":  round(min(scores), 4) if scores else 0.0,
            }
        return result

    def get_dashboard(self) -> Dict[str, Any]:
        """Full metrics snapshot for the admin endpoint."""
        with self._lock:
            uptime_s       = time.time() - self._startup_time
            req_count      = self._api_requests
            err_count      = self._api_errors
            total_lat      = self._total_latency
            models         = dict(self._models_loaded)
            last_err       = self._last_error
            sla_total      = self._sla_total
            sla_breached   = self._sla_breached

        avg_latency = (total_lat / req_count) if req_count else 0.0

        return {
            "uptime_seconds":     round(uptime_s, 1),
            "api": {
                "total_requests":   req_count,
                "errors":           err_count,
                "error_rate":       round(err_count / max(1, req_count), 4),
                "avg_latency_ms":   round(avg_latency * 1000, 2),
            },
            "models": {
                name: ("loaded" if status else "not_loaded")
                for name, status in models.items()
            },
            "sla": {
                "total_orders":   sla_total,
                "breached":       sla_breached,
                "breach_rate":    round(sla_breached / max(1, sla_total), 4),
                "target_breach_rate": 0.05,  # production target: < 5% SLA breaches
            },
            "last_error":         last_err,
            "episode_count":      len(self._episodes),
            "per_task_summary":   self.per_task_summary(),
        }


# ──────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────

metrics = MetricsStore()
