# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

import numpy as np
cimport numpy as cnp

from libc.math cimport pi, sin, cos


cdef:
    int NOOP = 0
    int FIRE = 1
    int LEFT = 2
    int RIGHT = 3

cdef class CBreakout:
    cdef:
        char[:] observations
        float[:] rewards
        float[:] ball_position
        float[:] ball_velocity
        float[:] paddle_position
        float[:, :] brick_positions
        float[:] brick_states
        float[:] wall_position
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
        self.wall_position = np.zeros(2, dtype=np.float32)

    cdef void compute_observations(self):
        cdef int i

        # Fix these casts so the data ranges make sense
        self.observations[0] = <char>(self.paddle_position[0])
        self.observations[1] = <char>(self.paddle_position[1])
        self.observations[2] = <char>(self.ball_position[0])
        self.observations[3] = <char>(self.ball_position[1])
        self.observations[4] = <char>(self.ball_velocity[0])
        self.observations[5] = <char>(self.ball_velocity[1])
        for i in range(self.num_bricks):
            self.observations[5 + i] = <char>(self.brick_states[i])

    def reset(self):
        self.paddle_position[0] = self.width / 2.0 - self.paddle_width // 2
        self.paddle_position[1] = self.height - self.paddle_height - 10

        self.ball_position[0] = self.paddle_position[0] + (self.paddle_width // 2 - self.ball_width // 2)
        self.ball_position[1] = self.paddle_position[1] - self.ball_height - 10

        self.ball_velocity[0] = 0.0
        self.ball_velocity[1] = 0.0

        for i in range(self.num_bricks):
            self.brick_states[i] = 0

        self.ball_fired = False
        self.score = 0

        self.compute_observations()

    def step(self, long[:] np_actions):
        cdef int action, i
        cdef float score, direction

        action = <int>np_actions[0]
        if action == NOOP:
            pass
        elif action == FIRE and not self.ball_fired:
            self.ball_fired = True

            direction = pi / 6

            self.ball_velocity[0] = (-sin(direction) if self.paddle_position[0] + self.paddle_width // 2 < self.width // 2
                                     else sin(direction)) * self.ball_speed

            self.ball_velocity[1] = -cos(direction) * self.ball_speed
        elif action == LEFT:
            if not self.paddle_position[0] <= 0:
                self.paddle_position[0] -= 2
        elif action == RIGHT:
            if not self.paddle_position[0] + self.paddle_width >= self.width:
                self.paddle_position[0] += 2

        if not self.ball_fired:
            self.ball_position[0] = self.paddle_position[0] + self.paddle_width // 2 - self.ball_width // 2

        self.handle_collisions()

        self.rewards[0] = self.score

        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1]

        if self.ball_position[1] >= self.paddle_position[1] + self.paddle_height or self.score == 448:
            self.done = True

        self.compute_observations()

    cdef void handle_collisions(self):
        cdef int i, row, velocity_index, wall_width, wall_height
        cdef float relative_intersection, direction

        # Left Wall
        self.wall_position[0] = 0.0
        wall_width = 0
        wall_height = self.height
        if self.check_collision(self.wall_position, wall_width, wall_height, self.ball_position, self.ball_width, self.ball_height):
            self.ball_velocity[0] *= -1

        # Top Wall
        wall_width = self.width
        wall_height = 0
        if self.check_collision(self.wall_position, wall_width, wall_height, self.ball_position, self.ball_width, self.ball_height):
            self.ball_velocity[1] *= -1

        # Right Wall
        self.wall_position[0] = self.width
        wall_width = 0
        wall_height = self.height
        if self.check_collision(self.wall_position, wall_width, wall_height, self.ball_position, self.ball_width, self.ball_height):
            self.ball_velocity[0] *= -1


        # Paddle Ball Collisions
        if self.check_collision(self.paddle_position, self.paddle_width, self.paddle_height, self.ball_position, self.ball_width, self.ball_height):

            relative_intersection = (self.paddle_position[0] + self.paddle_width//2) - (self.ball_position[0] + self.ball_width//2)
            relative_intersection = relative_intersection / (self.paddle_width // 2)
            direction = relative_intersection * pi/4

            self.ball_velocity[0] = sin(direction) * self.ball_speed
            self.ball_velocity[1] = -cos(direction) * self.ball_speed


        # Brick Ball Collisions
        for i in range(self.num_bricks):
            if self.brick_states[i] == 1:
                continue

            if self.check_collision(self.brick_positions[i], self.brick_width, self.brick_height, self.ball_position, self.ball_width, self.ball_height):
                self.brick_states[i] = 1

                # Determine collision direction and invert velocity accordingly
                if (self.ball_position[1] + self.ball_height <= self.brick_positions[i][1] + (self.brick_height / 2)) or \
                    (self.ball_position[1] >= self.brick_positions[i][1] + (self.brick_height / 2)):
                    # Vertical collision
                    self.ball_velocity[1] *= -1
                elif (self.ball_position[0] + self.ball_width <= self.brick_positions[i][0] + (self.brick_width / 2)) or \
                        (self.ball_position[0] >= self.brick_positions[i][0] + (self.brick_width / 2)):
                    # Horizontal collision
                    self.ball_velocity[0] *= -1

                row = i // self.num_brick_cols
                self.score += 7 - 2 * (row // 2)
                break


    cdef bint check_collision(self, float[:] rectangle_position, int rectangle_width, int rectangle_height, float[:] other_position, int other_width, int other_height):
        cdef float rectangle_left, rectangle_right, rectangle_top, rectangle_bottom, other_left, other_right, other_top, other_bottom

        rectangle_left = rectangle_position[0]
        rectangle_right = rectangle_position[0] + rectangle_width
        rectangle_top = rectangle_position[1]
        rectangle_bottom = rectangle_position[1] + rectangle_height

        other_left = other_position[0]
        other_right = other_position[0] + other_width
        other_top = other_position[1]
        other_bottom = other_position[1] + other_height

        return rectangle_right > other_left and rectangle_left < other_right and rectangle_bottom > other_top and rectangle_top < other_bottom
