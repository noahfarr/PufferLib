# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=True
# cython: initializedcheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: nonecheck=True
# cython: profile=False

cimport numpy as cnp

cdef class CBreakout:
    cdef:
        char[:] observations
        float[:] rewards
        float[:] ball_position
        float[:] ball_velocity
        float[:] paddle_position
        float[:, :] brick_positions
        float[:] brick_states
        int width
        int height
        int obs_size
        int num_bricks
        int num_brick_rows
        int num_brick_cols
        int paddle_width
        int paddle_height
        int ball_width
        int ball_height
        int brick_width
        int brick_height
        int score
        float ball_speed
        float brick_reward
        bint ball_fired
        bint done


    def __init__(self, cnp.ndarray observations, cnp.ndarray rewards, cnp.ndarray paddle_position, cnp.ndarray ball_position, cnp.ndarray ball_velocity, cnp.ndarray brick_positions, cnp.ndarray brick_states, bint done, int width, int height, int paddle_width, int paddle_height, int ball_width, int ball_height, int brick_width, int brick_height, int obs_size, int num_bricks, int num_brick_rows, int num_brick_cols, float brick_reward):
        self.observations = observations
        self.rewards = rewards
        self.paddle_position = paddle_position
        self.ball_position = ball_position
        self.ball_velocity = ball_velocity
        self.brick_positions = brick_positions
        self.brick_states = brick_states
        self.done = done
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.obs_size = obs_size
        self.num_bricks = num_bricks
        self.num_brick_rows = num_brick_rows
        self.num_brick_cols = num_brick_cols
        self.brick_reward = brick_reward
        self.ball_fired = False
        self.ball_speed = 4
        self.score = 0
