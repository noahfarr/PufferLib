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
        int ball_radius
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
            int ball_radius,
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
        self.ball_radius = ball_radius
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.brick_reward = brick_reward
        self.ball_fired = False
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

        self.ball_position[0] = self.paddle_position[0] + self.paddle_width // 2
        self.ball_position[1] = self.paddle_position[1] - self.paddle_height // 2 - self.ball_radius // 2

        self.ball_velocity[0] = 0.0
        self.ball_velocity[1] = 0.0

        for i in range(self.num_bricks):
            self.brick_states[i] = 0

        self.ball_fired = False
        self.score = 0

        self.compute_observations()

    def step(self, np_actions):
        cdef int action, i
        cdef float score

        action = np_actions[0]
        if action == NOOP:
            pass
        elif action == FIRE:
            if not self.ball_fired:
                self.ball_fired = True

                # TODO: Improve logic for where to shoot ball
                self.ball_velocity[0] = 2
                self.ball_velocity[1] = -2
        elif action == LEFT:
            self.paddle_position[0] -= 2
            pass
        elif action == RIGHT:
            self.paddle_position[0] += 2
            pass

        if not self.ball_fired:
            self.ball_position[0] = self.paddle_position[0] + self.paddle_width // 2

        self.handle_collisions()

        self.rewards[0] = self.score

        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1]

        self.compute_observations()

    cdef void handle_collisions(self):
        cdef int row
        self.handle_ball_wall_collisions()
        # Paddle Ball Collision
        if self.check_rectangle_circle_collision(self.paddle_position, self.paddle_width, self.paddle_height, self.ball_position, self.ball_radius):
            self.ball_position[1] = self.paddle_position[1] - self.ball_radius
            self.ball_velocity[1] *= -1

        # Ball Brick Collisions
        for i in range(self.num_bricks):
            if self.brick_states[i] == 1:
                continue

            if self.check_rectangle_circle_collision(self.brick_positions[i], self.brick_width, self.brick_height, self.ball_position, self.ball_radius):
                self.brick_states[i] = 1
                # TODO: Improve behaviour for ball brick hits from side
                self.ball_velocity[1] *= -1

                row = i // self.num_brick_cols
                self.score += 7 - 2 * (row // 2)
                break


    cdef bint check_rectangle_circle_collision(self, float[:] rectangle_position, int rectangle_width, int rectangle_height, float[:] circle_position, float circle_radius):
        cdef float rectangle_left, rectangle_right, rectangle_top, rectangle_bottom
        cdef float circle_left, circle_right, circle_top, circle_bottom
        cdef float closest_x, closest_y, dx, dy, squared_distance, squared_radius
        cdef bint overlaps

        # Rectangle coordinates
        rectangle_left = rectangle_position[0]
        rectangle_right = rectangle_position[0] + rectangle_width
        rectangle_top = rectangle_position[1]
        rectangle_bottom = rectangle_position[1] + rectangle_height

        # Circle bounding box coordinates
        circle_left = circle_position[0] - circle_radius
        circle_right = circle_position[0] + circle_radius
        circle_bottom = circle_position[1] + circle_radius
        circle_top = circle_position[1] - circle_radius

        # Check for overlap
        overlaps = (rectangle_right > circle_left and
                        rectangle_left < circle_right and
                        rectangle_top < circle_bottom and
                        rectangle_bottom > circle_top)

        if not overlaps:
            # print("Does not overlap...")
            return False


        # Find the closest point on the rectangle to the circle's center
        closest_x = max(rectangle_left, min(circle_position[0], rectangle_right))
        closest_y = max(rectangle_top, min(circle_position[1], rectangle_bottom))

        # Calculate squared distance from the closest point to the circle's center
        dx = circle_position[0] - closest_x
        dy = circle_position[1] - closest_y
        squared_distance = dx * dx + dy * dy
        squared_radius = circle_radius * circle_radius

        return squared_distance < squared_radius


    cdef void handle_ball_wall_collisions(self):
        # Collision with the left wall
        if self.ball_position[0] - self.ball_radius < 0:
            self.ball_position[0] = self.ball_radius  # Correct ball position after collision
            self.ball_velocity[0] *= -1

        # Collision with the right wall
        elif self.ball_position[0] + self.ball_radius > self.width:
            self.ball_position[0] = self.width - self.ball_radius  # Correct ball position after collision
            self.ball_velocity[0] *= -1

        # Collision with the top wall
        if self.ball_position[1] - self.ball_radius <= 0:
            self.ball_position[1] = self.ball_radius  # Correct ball position after collision
            self.ball_velocity[1] *= -1

        # Collision with the bottom wall
        elif self.ball_position[1] + self.ball_radius > self.height:
            self.done = True
            return
