"""Continuation experiment: measured red clocks and feasible action masking.

The 120 s bound applies to time without green, including yellow. The mask
reserves enough time to serve every other approach after the chosen action.
This is a controller constraint, not a learned safety/collision guarantee.
"""

import numpy as np
import torch
from .environment import Environment

MAX_RED = 120.0


def feasible_actions(ages, phases, limit=MAX_RED):
    """Return [batch, 4] feasibility, allowing a 3 s switch + 10 s green."""
    ages = np.atleast_2d(np.asarray(ages, dtype=float))
    phases = np.asarray(phases).reshape(-1)
    result = np.zeros((len(ages), 4), dtype=bool)
    for action in range(4):
        change = phases != action
        duration = 10.0 + 3.0 * change
        others = np.sort(np.delete(ages, action, axis=1), axis=1)[:, ::-1]
        # The selected phase starts now, or after yellow. Subsequently serve
        # the other approaches oldest first, each 13 s apart.
        selected_ok = ages[:, action] + 3.0 * change <= limit + 1e-6
        deadlines = others + duration[:, None] + 3.0 + 13.0 * np.arange(3)
        result[:, action] = selected_ok & (deadlines <= limit + 1e-6).all(1)
    return result


def state_masks(states):
    states = np.atleast_2d(states)
    ages = np.round(states[:, -4:] * MAX_RED * 2) / 2
    masks = feasible_actions(ages, states[:, 36:40].argmax(1))
    if not masks.any(1).all():
        raise RuntimeError("Red-time deadline is infeasible; refusing silent fallback")
    return torch.from_numpy(masks)


def masked_values(model, states):
    return model(states).masked_fill(
        ~state_masks(states.detach().cpu().numpy()), -torch.inf
    )


class ClockEnvironment(Environment):
    """Monitor clocks for all policies; enforce feasibility only when requested."""

    def __init__(self, *args, enforce=False, **kwargs):
        self.enforce = enforce
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.red_age = np.zeros(4)
        self.max_red = np.zeros(4)
        self.masked_decisions = 0
        self.active_green = None
        state = super().reset(*args, **kwargs)
        self.max_red = self.red_age.copy()
        return state

    def _signal(self, phase, yellow=False):
        super()._signal(phase, yellow)
        self.active_green = None if yellow else phase
        if not yellow:
            self.red_age[phase] = 0.0

    def _tick(self, measure=True):
        super()._tick(measure)
        self.red_age += 0.5
        if self.active_green is not None:
            self.red_age[self.active_green] = 0.0
        if measure:
            self.max_red = np.maximum(self.max_red, self.red_age)

    def state(self):
        return np.r_[super().state(), self.red_age / MAX_RED].astype(np.float32)

    def action_mask(self):
        mask = feasible_actions(self.red_age, self.phase)[0]
        if not mask.any():
            raise RuntimeError("No feasible action under configured red-time bound")
        return mask

    def step(self, action):
        if self.enforce:
            mask = self.action_mask()
            self.masked_decisions += int(not mask.all())
            if not mask[int(action)]:
                raise ValueError("Action would violate the red-time schedule")
        result = super().step(action)
        if self.record:
            self.trace[-1]["red_age"] = self.red_age.tolist()
        return result

    def metrics(self):
        return {
            **super().metrics(),
            "max_without_green_s": float(self.max_red.max()),
            "max_without_green_by_approach_s": self.max_red.tolist(),
            "constrained_decisions": self.masked_decisions,
            "red_limit_s": MAX_RED if self.enforce else None,
        }
