import gymnasium

import numpy as np
from numpy._typing import _NumberLike_co
from yaml.events import NodeEvent

from raylib import rl, colors

import pufferlib
from pufferlib.environments.ocean import render
from pufferlib.environments.ocean.breakout.c_breakout import CBreakout


class PufferBreakout(pufferlib.PufferEnv):

    def __init__(
        self,
        width: int = 480,
        height: int = 576,
        num_brick_rows: int = 8,
        num_brick_cols: int = 14,
        brick_reward: float = 1,
        report_interval: int = 128,
        render_mode: str = "rgb_array",
    ) -> None:
        self.width = width
        self.height = height
        self.grid = [np.zeros((height, width), dtype=np.uint8)]
        self.num_brick_rows = num_brick_rows
        self.num_brick_cols = num_brick_cols
        self.num_bricks = num_brick_rows * num_brick_cols
        self.brick_reward = brick_reward
        self.report_interval = report_interval

        self.paddle_position = np.zeros(2, dtype=np.float32)
        self.ball_position = np.zeros(2, dtype=np.float32)
        self.ball_velocity = np.zeros(2, dtype=np.float32)
        self.brick_states = np.zeros(self.num_bricks, np.float32)

        self.c_env: CBreakout | None = None
        self.human_action = None
        self.tick = 0
        self.reward_sum = 0

        self.paddle_width: int = 50
        self.paddle_height: int = 8
        self.ball_width: int = 8
        self.ball_height: int = 8
        self.brick_width: int = self.width // self.num_brick_cols
        self.brick_height: int = 8
        self.brick_positions = self.generate_brick_positions()

        # This block required by advanced PufferLib env spec
        self.obs_size = 2 + 2 + self.num_bricks
        low = np.array([0.0] * 2 + [0.0] * 2 + [0.0] * 2 + [0.0] * self.num_bricks)
        high = np.array(
            [width, height] + [width, height] + [2.0] * 2 + [1.0] * self.num_bricks
        )
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(4)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1
        self.render_mode = render_mode
        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations=np.zeros(
                (self.num_agents, *self.observation_space.shape), dtype=np.uint8
            ),
            rewards=np.zeros(self.num_agents, dtype=np.float32),
            terminals=np.zeros(self.num_agents, dtype=bool),
            truncations=np.zeros(self.num_agents, dtype=bool),
            masks=np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros(self.num_agents, dtype=np.uint32)

        if render_mode == "ansi":
            self.client = render.AnsiRender()
        elif render_mode == "rgb_array":
            self.client = render.RGBArrayRender()
        elif render_mode == "human":
            self.client = RaylibClient(
                self.width,
                self.height,
                self.num_brick_rows,
                self.num_brick_cols,
                self.brick_positions,
                self.paddle_width,
                self.paddle_height,
                self.ball_width,
                self.ball_height,
                self.brick_width,
                self.brick_height,
            )
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def step(self, actions):
        if self.render_mode == "human" and self.human_action is not None:
            actions[0] = self.human_action

        self.actions = actions
        self.c_env.step(actions)

        if (
            self.ball_position[1] > self.paddle_position[1] + self.paddle_height
            or self.buf.rewards[0] == 448
        ):
            self.done = True

        info = {}
        self.reward_sum += self.buf.rewards.mean()

        if self.tick % self.report_interval == 0:
            info = {"rewards": self.reward_sum / self.report_interval}
            self.reward_sum = 0
        return (
            self.buf.observations,
            self.buf.rewards,
            self.buf.terminals,
            self.buf.truncations,
            info,
        )

    def reset(self, seed=None):
        if self.c_env is None:
            self.c_env = CBreakout(
                observations=self.buf.observations[0],
                rewards=self.buf.rewards,
                done=self.done,
                paddle_position=self.paddle_position,
                ball_position=self.ball_position,
                ball_velocity=self.ball_velocity,
                brick_positions=self.brick_positions,
                brick_states=self.brick_states,
                width=self.width,
                height=self.height,
                obs_size=self.obs_size,
                num_bricks=self.num_bricks,
                num_brick_rows=self.num_brick_rows,
                num_brick_cols=self.num_brick_cols,
                paddle_width=self.paddle_width,
                paddle_height=self.paddle_height,
                ball_width=self.ball_width,
                ball_height=self.ball_height,
                brick_width=self.brick_width,
                brick_height=self.brick_height,
                brick_reward=self.brick_reward,
            )

        self.done = False
        self.c_env.reset()
        return self.buf.observations, {}

    def render(self):
        if self.render_mode == "ansi":
            return self.client.render(self.grid)
        elif self.render_mode == "rgb_array":
            return self.client.render(self.grid)
        elif self.render_mode == "raylib":
            return self.client.render(self.grid)
        elif self.render_mode == "human":
            action = None

            if rl.IsKeyDown(rl.KEY_LEFT) or rl.IsKeyDown(rl.KEY_A):
                action = 2
            if rl.IsKeyDown(rl.KEY_RIGHT) or rl.IsKeyDown(rl.KEY_D):
                action = 3

            self.human_action = action

            return self.client.render(
                self.paddle_position,
                self.ball_position,
                self.brick_states,
                self.buf.rewards[0],
            )
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def close(self):
        pass

    def generate_brick_positions(self):
        brick_positions = np.zeros((self.num_bricks, 2), dtype=np.float32)
        y_offset = 30
        for row in range(self.num_brick_rows):
            for col in range(self.num_brick_cols):
                idx = row * self.num_brick_cols + col
                x = col * self.brick_width
                y = row * self.brick_height + y_offset
                brick_positions[idx][0] = x
                brick_positions[idx][1] = y
        return brick_positions


class RaylibClient:
    def __init__(
        self,
        width: int,
        height: int,
        num_brick_rows: int,
        num_brick_cols: int,
        brick_positions: np.ndarray,
        paddle_width: int,
        paddle_height: int,
        ball_width: float,
        ball_height: float,
        brick_width: int,
        brick_height: int,
    ) -> None:
        self.width = width
        self.height = height
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.num_brick_rows = num_brick_rows
        self.num_brick_cols = num_brick_cols
        self.brick_positions = brick_positions

        self.running = False

        self.BRICK_COLORS = [
            colors.RED,
            colors.RED,
            colors.ORANGE,
            colors.ORANGE,
            colors.GREEN,
            colors.GREEN,
            colors.YELLOW,
            colors.YELLOW,
        ]

        # Initialize raylib window
        rl.InitWindow(width, height, "PufferLib Ray Breakout".encode())
        rl.SetTargetFPS(60)

    def render(
        self,
        paddle_position: np.ndarray,
        ball_position: np.ndarray,
        brick_states: np.ndarray,
        score: float,
    ) -> None:
        rl.BeginDrawing()
        rl.ClearBackground(render.PUFF_BACKGROUND)

        # Draw the paddle
        paddle_x, paddle_y = paddle_position
        rl.DrawRectangle(
            int(paddle_x),
            int(paddle_y),
            self.paddle_width,
            self.paddle_height,
            colors.DARKGRAY,
        )

        # Draw the ball
        ball_x, ball_y = ball_position
        rl.DrawRectangle(
            int(ball_x), int(ball_y), self.ball_width, self.ball_height, colors.WHITE
        )

        # Draw the bricks
        for row in range(self.num_brick_rows):
            for col in range(self.num_brick_cols):
                idx = row * self.num_brick_cols + col
                if brick_states[idx] == 1:
                    continue

                x, y = self.brick_positions[idx]
                brick_color = self.BRICK_COLORS[row]
                rl.DrawRectangle(
                    int(x),
                    int(y),
                    self.brick_width,
                    self.brick_height,
                    brick_color,
                )

        # Draw Score
        text = f"Score: {int(score)}".encode("ascii")
        rl.DrawText(text, 10, 10, 20, colors.WHITE)
        rl.EndDrawing()

    def close(self) -> None:
        rl.close_window()
