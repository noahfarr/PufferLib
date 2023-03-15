from pdb import set_trace as T
import numpy as np

import functools

import gym
from pettingzoo.utils.env import ParallelEnv

import pufferlib


class TestEnv(ParallelEnv):
    '''
    A complex testing environment with:
        - Multiple and variable agent population
        - Hierarchical observation and action spaces
    '''
    def __init__(self, initial_agents=1, max_agents=100,
                 spawn_attempts_per_tick=2, death_per_tick=1):

        assert death_per_tick <= initial_agents

        self.possible_agents = [abs(hash(str(i))) % 2**16 for i in range(max_agents)]

        self.initial_agents = initial_agents
        self.max_agents = max_agents
        self.spawn_attempts_per_tick = spawn_attempts_per_tick
        self.death_per_tick = death_per_tick
        self.rng = None

    def seed(self, seed):
        self.rng = pufferlib.utils.RandomState(seed)

    def reset(self):
        self.tick = 0
        assert self.rng is not None, 'Must seed environment before reset'
        self.agents = self.rng.sample(self.possible_agents, self.initial_agents)
        return {a: self._observation(a) for a in self.agents}

    def step(self, action):
        obs, rewards, dones, infos = {}, {}, {}, {}

        dead  = self.rng.sample(self.agents, self.death_per_tick)
        for kill in dead:
            self.agents.remove(kill)
            obs[kill] = self._observation(kill)
            rewards[kill] = -1
            dones[kill] = True
            infos[kill] = {'dead': True}

        for spawn in range(self.spawn_attempts_per_tick):
            spawn = self.rng.choice(self.possible_agents)
            if spawn not in self.agents + dead:
                self.agents.append(spawn)

        for agent in self.agents:
            obs[agent] = self._observation(spawn)
            rewards[agent] = 0.1 * self.rng.random()
            dones[agent] = False
            infos[agent] = {'dead': False}

        self.tick += 1
        return obs, rewards, dones, infos

    def _observation(self, agent):
        return {
            'foo': np.arange(23, dtype=np.float32) + agent,
            'bar': np.arange(45, dtype=np.float32) + agent,
            'baz': {
                'qux': np.arange(6, dtype=np.float32) + agent,
                'quux': np.arange(7, dtype=np.float32) + agent,
            }
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Dict({
            'foo': gym.spaces.Box(low=0, high=1, shape=(23,)),
            'bar': gym.spaces.Box(low=0, high=1, shape=(45,)),
            'baz': gym.spaces.Dict({
                'qux': gym.spaces.Box(low=0, high=1, shape=(6,)),
                'quux': gym.spaces.Box(low=0, high=1, shape=(7,)),
            })
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Dict({
            'foo': gym.spaces.Discrete(2),
            'bar': gym.spaces.Dict({
                'baz': gym.spaces.Discrete(7),
                'qux': gym.spaces.Discrete(8),
            })
        })

    def pack_ob(self, ob):
        # Note: there's currently a weird sort order on obs
        return np.concatenate([ob['bar'], ob['foo'],  ob['baz']['quux'], ob['baz']['qux']])