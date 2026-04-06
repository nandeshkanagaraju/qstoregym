"""
Curriculum learning wrapper for QStoreGymWrapper.

Previously, CurriculumWrapper wrapped QStoreEnv directly with a non-Gymnasium interface,
making it impossible to use with SB3. This rewrite wraps QStoreGymWrapper so it exposes
a valid Gymnasium Env interface, and changes tasks in-place when the agent earns promotion.

A single model trains progressively across all tasks rather than training 5 separate
models that never share knowledge.

Promotion rule: achieve score >= promotion_threshold on consecutive_required consecutive
episodes before advancing to the next task.
"""
from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_wrapper import QStoreGymWrapper, OBS_DIM, ACT_DIM
from tasks import AVAILABLE_TASKS


class CurriculumGymWrapper(gym.Env):
    """
    Gymnasium-compatible curriculum wrapper.

    Delegates all Gymnasium calls to an underlying QStoreGymWrapper instance.
    At the end of each episode (when `terminated` is True), checks the episode
    score against the promotion threshold. Advances task when the agent achieves
    `consecutive_required` consecutive promotable scores.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        promotion_threshold: float = 0.6,
        consecutive_required: int = 3,
        start_task_idx: int = 0,
    ):
        super().__init__()

        # Validate start
        assert 0 <= start_task_idx < len(AVAILABLE_TASKS), "Invalid start_task_idx"
        self.current_task_idx = start_task_idx
        self.promotion_threshold = promotion_threshold
        self.consecutive_required = consecutive_required

        self._build_inner_env()

        # Mirror the inner env's spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._recent_scores: List[float] = []
        self._episode_count = 0
        self._last_episode_score: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_inner_env(self):
        task_name = AVAILABLE_TASKS[self.current_task_idx]
        self._inner: QStoreGymWrapper = QStoreGymWrapper(task_name=task_name)

    def _maybe_promote(self):
        if len(self._recent_scores) >= self.consecutive_required:
            window = self._recent_scores[-self.consecutive_required:]
            if all(s >= self.promotion_threshold for s in window):
                if self.current_task_idx < len(AVAILABLE_TASKS) - 1:
                    self.current_task_idx += 1
                    self._recent_scores = []
                    self._build_inner_env()
                    print(
                        f"\n*** Curriculum Promotion! "
                        f"Advanced to '{AVAILABLE_TASKS[self.current_task_idx]}' "
                        f"(task {self.current_task_idx + 1}/{len(AVAILABLE_TASKS)}) ***\n"
                    )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        obs, info = self._inner.reset(seed=seed, options=options)
        info["task"] = AVAILABLE_TASKS[self.current_task_idx]
        info["task_idx"] = self.current_task_idx
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._inner.step(action)

        if terminated or truncated:
            episode_score = info.get("score", 0.0)
            self._last_episode_score = episode_score
            self._recent_scores.append(episode_score)
            self._episode_count += 1
            self._maybe_promote()

        info["task"] = AVAILABLE_TASKS[self.current_task_idx]
        info["task_idx"] = self.current_task_idx
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Properties for monitoring
    # ------------------------------------------------------------------

    @property
    def current_task_name(self) -> str:
        return AVAILABLE_TASKS[self.current_task_idx]

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def last_episode_score(self) -> float:
        return self._last_episode_score
