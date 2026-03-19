from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from slimevolleygym import slimevolley as sv


@dataclass
class HeuristicConfig:
    defend_x: float = 14.0
    intercept_x: float = 10.0
    retreat_x: float = 18.5
    deep_retreat_x: float = 18.0
    edge_recenter_x: float = 15.2
    deep_ball_x: float = 17.5
    falling_ball_x: float = 12.5
    falling_ball_vy: float = 1.5
    general_under_ball_bias: float = 0.9
    under_ball_bias: float = 0.6
    min_x: float = 8.5
    max_x: float = 22.0
    ball_gravity: float = sv.GRAVITY
    max_prediction_steps: int = 45
    jump_cooldown_steps: int = 8
    timestep: float = sv.TIMESTEP
    floor_y: float = sv.REF_U
    court_half_width: float = sv.REF_W / 2.0
    high_ball_y: float = 13.0
    edge_ball_x: float = 18.2
    low_horizontal_speed: float = 6.0


class HeuristicPolicy:
    def __init__(self, config: HeuristicConfig | None = None):
        self.config = config or HeuristicConfig()
        self.jump_cooldown = 0

    def reset(self):
        self.jump_cooldown = 0

    def predict(self, obs):
        x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, _op_x, _op_y, _op_vx, _op_vy = [
            float(value) * 10.0 for value in obs
        ]

        landing_x = self._predict_landing_x(ball_x, ball_y, ball_vx, ball_vy)
        ball_on_our_side = ball_x > 0.0
        ball_high = ball_y > self.config.high_ball_y
        ball_deep_on_our_side = ball_on_our_side and ball_x > self.config.deep_ball_x
        ball_high_near_edge = (
            ball_on_our_side
            and ball_high
            and ball_x > self.config.edge_ball_x
            and abs(ball_vx) < self.config.low_horizontal_speed
        )
        ball_falling_on_our_side = (
            ball_on_our_side
            and ball_x > self.config.falling_ball_x
            and ball_vy < self.config.falling_ball_vy
        )
        ball_crossing_soon = ball_vx > 0.0 and ball_x > -4.0
        reachable_landing = landing_x >= self.config.min_x

        if ball_on_our_side and reachable_landing:
            target_x = np.clip(landing_x, self.config.min_x, self.config.max_x)
        elif ball_crossing_soon and not ball_high:
            target_x = self.config.intercept_x
        elif ball_high_near_edge:
            target_x = self.config.edge_recenter_x
        elif ball_high:
            target_x = self.config.deep_retreat_x
        else:
            target_x = self.config.retreat_x

        if ball_deep_on_our_side:
            under_ball_target = ball_x - self.config.under_ball_bias
            target_x = np.clip(
                min(target_x, under_ball_target), self.config.min_x, self.config.max_x
            )
        elif ball_falling_on_our_side:
            under_ball_target = ball_x - self.config.general_under_ball_bias
            target_x = np.clip(
                min(target_x, under_ball_target), self.config.min_x, self.config.max_x
            )

        forward = x > target_x + 0.75
        backward = x < target_x - 0.75

        if x < self.config.min_x and not ball_on_our_side:
            forward = False
            backward = True

        ball_close_x = abs(ball_x - x) < 2.6
        ball_close_y = y + 3.0 < ball_y < y + 11.5
        ball_descending = ball_vy < 2.0
        attack_jump = (
            ball_on_our_side and ball_close_x and ball_close_y and ball_descending
        )
        if ball_deep_on_our_side:
            under_ball_ready = x >= ball_x - 1.6 and x <= ball_x + 0.4
            attack_jump = attack_jump and under_ball_ready and ball_vy < 0.5
        elif ball_falling_on_our_side:
            under_ball_ready = x >= ball_x - 2.0 and x <= ball_x + 0.2
            attack_jump = attack_jump and under_ball_ready and ball_vy < 1.0

        emergency_jump = (
            ball_on_our_side
            and ball_y < 5.5
            and abs(ball_x - x) < 2.2
            and ball_vy < 0.8
        )
        recovery_jump = (
            y < 1.8
            and abs(vx) < 0.5
            and ball_on_our_side
            and ball_y > 8.0
            and abs(ball_x - x) < 4.0
        )
        jump = False
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        elif attack_jump or emergency_jump or recovery_jump:
            jump = True
            self.jump_cooldown = self.config.jump_cooldown_steps

        return [int(forward), int(backward), int(jump)]

    def _predict_landing_x(
        self, ball_x: float, ball_y: float, ball_vx: float, ball_vy: float
    ) -> float:
        predicted_x = ball_x
        predicted_y = ball_y
        velocity_x = ball_vx
        velocity_y = ball_vy

        for _ in range(self.config.max_prediction_steps):
            predicted_x += velocity_x * self.config.timestep
            predicted_y += velocity_y * self.config.timestep
            velocity_y += self.config.ball_gravity * self.config.timestep

            if predicted_x < -self.config.court_half_width:
                predicted_x = -2.0 * self.config.court_half_width - predicted_x
                velocity_x *= -1.0
            elif predicted_x > self.config.court_half_width:
                predicted_x = 2.0 * self.config.court_half_width - predicted_x
                velocity_x *= -1.0

            if predicted_y <= self.config.floor_y:
                break

        return predicted_x
