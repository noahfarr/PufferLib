'''High-perf many-agent snake. Inspired by snake env from https://github.com/dnbt777'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.snake.cy_snake import Snake as CSnake

class Snake(pufferlib.PufferEnv):
    def __init__(self, widths=[2560], heights=[1440], num_snakes=[4096],
            num_food=[65536], vision=5, leave_corpse_on_death=True,
            reward_food=0.1, reward_corpse=0.1, reward_death=-1.0,
            report_interval=128,  max_snake_length=1024,
            render_mode='rgb_array', buf=None):

        assert isinstance(vision, int)
        if isinstance(leave_corpse_on_death, bool):
            leave_corpse_on_death = len(widths)*[leave_corpse_on_death]

        assert (len(widths) == len(heights) == len(num_snakes)
            == len(num_food) == len(leave_corpse_on_death))

        for w, h in zip(widths, heights):
            assert w >= 2*vision+2 and h >= 2*vision+2, \
                'width and height must be at least 2*vision+2'

        total_snakes = sum(num_snakes)
        max_area = max([w*h for h, w in zip(heights, widths)])
        self.max_snake_length = min(max_snake_length, max_area)
        snake_shape = (total_snakes, self.max_snake_length, 2)
        self.report_interval = report_interval

        # This block required by advanced PufferLib env spec
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(2*vision+1, 2*vision+1), dtype=np.int8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.num_agents = total_snakes
        self.render_mode = render_mode
        self.tick = 0

        self.cell_size = int(np.ceil(1280 / max(max(widths), max(heights))))

        super().__init__(buf)
        self.c_envs = CSnake(self.observations, self.actions,
            self.rewards, self.terminals, widths, heights,
            num_snakes, num_food, vision, max_snake_length,
            leave_corpse_on_death, reward_food, reward_corpse,
            reward_death)
 
    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render(self.cell_size)

def test_performance(timeout=10, atn_cache=1024):
    env = Snake()
    env.reset()
    tick = 0

    total_snakes = sum(env.num_snakes)
    actions = np.random.randint(0, 4, (atn_cache, total_snakes))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', total_snakes * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()