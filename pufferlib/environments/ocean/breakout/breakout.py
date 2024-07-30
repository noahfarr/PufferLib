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
        width: int = 1280,
        height: int = 720,
        horizon: int = 100,
        num_bricks: int = 112,
        render_mode: str = "human",
    ) -> None:

        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.horizon = horizon
        self.num_bricks = num_bricks
        self.paddle_position = np.zeros(2)
        self.ball_position = np.zeros(2)
        self.brick_states = np.zeros(num_bricks)
        self.c_env: CBreakout | None = None
        self.human_action = None

        # This block required by advanced PufferLib env spec
        self.obs_size = 2 + 2 + num_bricks
        low = np.array([0.0] * 2 + [0.0] * 2 + [0.0] * num_bricks)
        high = np.array([width, height] + [width, height] + [1.0] * num_bricks)
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
            self.client = RaylibClient(self.width, self.height)
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def step(self, actions):
        if self.render_mode == "human" and self.human_action is not None:
            actions[0] = self.human_action

        self.actions = actions
        self.c_env.step(actions)
        infos = {}
        return (
            self.buf.observations,
            self.buf.rewards,
            self.buf.terminals,
            self.buf.truncations,
            infos,
        )

    def reset(self, seed=None):
        if self.c_env is None:
            self.c_env = CBreakout(
                observations=self.buf.observations,
                rewards=self.buf.rewards,
                paddle_position=self.paddle_position,
                ball_position=self.ball_position,
                brick_states=self.brick_states,
                width=self.width,
                height=self.height,
                horizon=self.horizon,
                obs_size=self.obs_size,
            )

        self.done = False
        self.c_env.reset(seed=seed)
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
                self.paddle_position, self.ball_position, self.brick_states
            )
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def close(self):
        pass


class RaylibClient:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ball_radius = 5
        self.paddle_width = 60
        self.paddle_height = 10
        self.brick_width = 50
        self.brick_height = 20
        self.running = False

        # Initialize raylib window
        rl.InitWindow(width, height, "PufferLib Ray Breakout".encode())
        rl.SetTargetFPS(60)

    def render(
        self,
        paddle_position: np.ndarray,
        ball_position: np.ndarray,
        brick_states: np.ndarray,
    ) -> None:
        rl.BeginDrawing()
        rl.ClearBackground(render.PUFF_BACKGROUND)

        # Draw the paddle
        paddle_x, paddle_y = paddle_position
        rl.DrawRectangle(
            int(paddle_x - self.paddle_width // 2),
            int(self.height - self.paddle_height - 10),
            self.paddle_width,
            self.paddle_height,
            colors.DARKGRAY,
        )

        # Draw the ball
        ball_x, ball_y = ball_position
        rl.DrawCircle(int(ball_x), int(ball_y), self.ball_radius, colors.WHITE)

        # # Draw the bricks
        # rows = 8
        # cols = 14
        # for row in range(rows):
        #     for col in range(cols):
        #         state = brick_states[(row * cols + col)]
        #         rl.DrawRectangle(
        #             col * self.brick_width,
        #             row * self.brick_height,
        #             self.brick_width,
        #             self.brick_height,
        #             colors.RED,
        #         )

        rl.EndDrawing()

    def close(self) -> None:
        rl.close_window()
