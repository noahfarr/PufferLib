# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=True
# cython: initializedcheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: nonecheck=True
# cython: profile=False

from libc.stdlib cimport rand
import numpy as np
cimport numpy as cnp

cdef:
    int NOOP = 0
    int FIRE = 1
    int LEFT = 2
    int RIGHT = 3

cdef class CBreakout:
    cdef:
        int width
        int height
        int horizon
        int obs_size

        unsigned char[:] observations
        float[:] rewards
        float[:] ball_position
        float[:] paddle_position
        float[:] brick_states
        int num_brick_states


    def __init__(self, cnp.ndarray observations, cnp.ndarray rewards, cnp.ndarray paddle_position, cnp.ndarray ball_position, cnp.ndarray brick_states, int width, int height, int horizon, int obs_size):
        self.width = width
        self.height = height
        self.horizon = horizon
        self.obs_size = obs_size
        self.observations = observations
        self.rewards = rewards
        self.paddle_position = paddle_position
        self.ball_position = ball_position
        self.brick_states = brick_states
        self.num_brick_states = brick_states.shape[0]

    cdef void compute_observations(self):
        # Fix these casts so the data ranges make sense
        self.observations[0] = <unsigned char>(self.paddle_position[0])
        self.observations[1] = <unsigned char>(self.paddle_position[1])
        self.observations[2] = <unsigned char>(self.ball_position[0])
        self.observations[4] = <unsigned char>(self.ball_position[1])
        #cdef int i
        #for i in range(self.num_brick_states):
        #    self.observations[3 + i] = <unsigned char>(self.brick_states[i])

    def reset(self, seed=0):
        self.paddle_position[0] = self.width / 2.0
        self.compute_observations()

    def step(self, np_actions):
        cdef int action
        action = np_actions[0]
        if action == NOOP:
            pass
        elif action == FIRE:
            pass
        elif action == LEFT:
            self.paddle_position[0] -= 1
        elif action == RIGHT:
            self.paddle_position[0] += 1
        self.compute_observations()
