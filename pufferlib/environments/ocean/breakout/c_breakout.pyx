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

cdef extern from "math.h":
    double sqrt(double x) nogil


cdef class CBreakout:
    cdef:
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
        float ball_speed
        int brick_width
        int brick_height
        float brick_reward
        float score
        bint ball_fired
        bint done

        unsigned char[:] observations
        float[:] rewards
        float[:] ball_position
        float[:] ball_velocity
        float[:] paddle_position
        float[:, :] brick_positions
        float[:] brick_states


    def __init__(self,
            cnp.ndarray observations,
            cnp.ndarray rewards,
            cnp.ndarray paddle_position,
            cnp.ndarray ball_position,
            cnp.ndarray ball_velocity,
            cnp.ndarray brick_positions,
            cnp.ndarray brick_states,
            bint done,
            int width,
            int height,
            int paddle_width,
            int paddle_height,
            int ball_width,
            int ball_height,
            int brick_width,
            int brick_height,
            int obs_size,
            int num_bricks,
            int num_brick_rows,
            int num_brick_cols,
            float brick_reward,
        ):
        self.width = width
        self.height = height
        self.obs_size = obs_size
        self.observations = observations
        self.rewards = rewards
        self.done = done
        self.paddle_position = paddle_position
        self.ball_position = ball_position
        self.ball_velocity = ball_velocity
        self.brick_positions = brick_positions
        self.brick_states = brick_states
        self.num_bricks = num_bricks
        self.num_brick_rows = num_brick_rows
        self.num_brick_cols = num_brick_cols
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.brick_reward = brick_reward
        self.ball_fired = False
        self.ball_speed = 3
        self.score = 0.0

    cdef void compute_observations(self):
        cdef int i

        # Fix these casts so the data ranges make sense
        self.observations[0] = <unsigned char>(self.paddle_position[0])
        self.observations[1] = <unsigned char>(self.paddle_position[1])
        self.observations[2] = <unsigned char>(self.ball_position[0])
        self.observations[3] = <unsigned char>(self.ball_position[1])
        self.observations[4] = <unsigned char>(self.ball_velocity[0])
        self.observations[5] = <unsigned char>(self.ball_velocity[1])
        for i in range(self.num_bricks):
            self.observations[5 + i] = <unsigned char>(self.brick_states[i])

    def reset(self, seed=0):
        self.paddle_position[0] = self.width / 2.0 - self.paddle_width // 2
        self.paddle_position[1] = self.height - self.paddle_height - 10

        self.ball_position[0] = self.paddle_position[0] + (self.paddle_width // 2 - self.ball_width // 2)
        self.ball_position[1] = self.paddle_position[1] - self.ball_height

        self.ball_velocity[0] = 0.0
        self.ball_velocity[1] = 0.0

        for i in range(self.num_bricks):
            self.brick_states[i] = 0

        self.ball_fired = False
        self.score = 0

        self.compute_observations()

    def step(self, np_actions):
        cdef int action, i
        cdef float score, angle

        action = np_actions[0]
        if action == NOOP:
            pass
        elif action == FIRE:
            if not self.ball_fired:
                self.ball_fired = True

                # TODO: Improve logic for where to shoot ball
                angle = np.pi / 6
                self.ball_velocity[0] = np.cos(angle) * self.ball_speed * (-1 if (self.paddle_position[0] + self.paddle_width//2 < self.width//2) else 1)
                self.ball_velocity[1] = np.sin(angle) * self.ball_speed
        elif action == LEFT:
            if not self.paddle_position[0] <= 0:
                self.paddle_position[0] -= 2
        elif action == RIGHT:
            if not self.paddle_position[0] + self.paddle_width >= self.width:
                self.paddle_position[0] += 2

        if not self.ball_fired:
            self.ball_position[0] = self.paddle_position[0] + self.paddle_width // 2

        self.handle_collisions()

        self.rewards[0] = self.score

        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1]

        self.compute_observations()

    cdef void handle_collisions(self):
        cdef int row, velocity_index, wall_width, wall_height
        cdef float[:, :] wall_positions
        cdef float[:] wall_position

        cdef int[:] wall_widths, wall_heights, velocity_indices

        wall_positions = np.array([[0.0, 0.0], [0.0, 0.0], [self.width, 0.0]], dtype=np.float32)
        wall_widths = np.array([0, self.width, 0], dtype=np.int32)
        wall_heights = np.array([self.height, 0, self.height], dtype=np.int32)
        velocity_indices = np.array([0, 1, 0], dtype=np.int32)

        # Wall Ball Collisions
        for i in range(3):
            wall_position = wall_positions[i]
            wall_width = wall_widths[i]
            wall_height = wall_heights[i]
            velocity_index = velocity_indices[i]
            if self.check_collision(wall_position, wall_width, wall_height, self.ball_position, self.ball_width, self.ball_height):
                self.ball_velocity[velocity_index] *= -1


        # Paddle Ball Collisions
        if self.check_collision(self.paddle_position, self.paddle_width, self.paddle_height, self.ball_position, self.ball_width, self.ball_height):
            # TODO: Improve collision resolution
            self.ball_velocity[1] *= -1

        # Brick Ball Collisions
        for i in range(self.num_bricks):
            if self.brick_states[i] == 1:
                continue

            if self.check_collision(self.brick_positions[i], self.brick_width, self.brick_height, self.ball_position, self.ball_width, self.ball_height):
                self.brick_states[i] = 1
                # TODO: Improve collision resolution
                self.ball_velocity[1] *= -1

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
