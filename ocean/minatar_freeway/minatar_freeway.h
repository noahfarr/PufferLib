#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define MINATAR_GRID 10
#define MINATAR_CHANNELS 7
#define MINATAR_OBS_SIZE (MINATAR_GRID*MINATAR_GRID*MINATAR_CHANNELS)

#define PLAYER_SPEED 3
#define TIME_LIMIT 2500
#define NUM_CARS 8

#define ACT_NOOP  0
#define ACT_LEFT  1
#define ACT_UP    2
#define ACT_RIGHT 3
#define ACT_DOWN  4
#define ACT_FIRE  5

#define CH_CHICKEN 0
#define CH_CAR     1
#define CH_SPEED1  2
#define CH_SPEED2  3
#define CH_SPEED3  4
#define CH_SPEED4  5
#define CH_SPEED5  6

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int x;
    int y;
    int countdown;
    int speed;
} Car;

typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    int pos;
    int move_timer;
    int terminate_timer;
    Car cars[NUM_CARS];
    int terminal;

    float episode_return;
    int episode_length;
    int max_steps;

    float sticky_action_prob;
    int ramping;
    int last_action;

    unsigned int rng;
} MinatarFreeway;

static inline float rndf(MinatarFreeway* env) {
    return (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
}

void init(MinatarFreeway* env) {

    (void)env;
}

void c_close(MinatarFreeway* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

static void randomize_cars(MinatarFreeway* env, int initialize) {
    int speeds[NUM_CARS];
    for (int i = 0; i < NUM_CARS; i++) {

        int speed = 1 + (rand_r(&env->rng) % 5);

        int direction = (rndf(env) - 0.5f < 0.0f) ? -1 : 1;
        speeds[i] = speed * direction;
    }
    if (initialize) {
        for (int i = 0; i < NUM_CARS; i++) {
            env->cars[i].x = 0;
            env->cars[i].y = i + 1;
            env->cars[i].countdown = abs(speeds[i]);
            env->cars[i].speed = speeds[i];
        }
    } else {
        for (int i = 0; i < NUM_CARS; i++) {
            env->cars[i].countdown = abs(speeds[i]);
            env->cars[i].speed = speeds[i];
        }
    }
}

void compute_observations(MinatarFreeway* env) {
    memset(env->observations, 0, MINATAR_OBS_SIZE);
    unsigned char* o = env->observations;

    o[(env->pos*MINATAR_GRID + 4)*MINATAR_CHANNELS + CH_CHICKEN] = 1;

    for (int i = 0; i < NUM_CARS; i++) {
        Car* car = &env->cars[i];

        o[(car->y*MINATAR_GRID + car->x)*MINATAR_CHANNELS + CH_CAR] = 1;

        int back_x = (car->speed > 0) ? car->x - 1 : car->x + 1;
        if (back_x < 0) back_x = 9;
        else if (back_x > 9) back_x = 0;

        int trail;
        int abs_speed = abs(car->speed);
        if (abs_speed == 1)      trail = CH_SPEED1;
        else if (abs_speed == 2) trail = CH_SPEED2;
        else if (abs_speed == 3) trail = CH_SPEED3;
        else if (abs_speed == 4) trail = CH_SPEED4;
        else                     trail = CH_SPEED5;

        o[(car->y*MINATAR_GRID + back_x)*MINATAR_CHANNELS + trail] = 1;
    }
}

void c_reset(MinatarFreeway* env) {
    randomize_cars(env, 1);
    env->pos = 9;
    env->move_timer = PLAYER_SPEED;
    env->terminate_timer = TIME_LIMIT;
    env->terminal = 0;

    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->last_action = ACT_NOOP;
    compute_observations(env);
}

void add_log(MinatarFreeway* env) {
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->episode_return;
    env->log.score += env->episode_return;
    env->log.perf += env->episode_return;
    env->log.n += 1.0f;
}

static float act(MinatarFreeway* env, int a) {
    float r = 0.0f;
    if (env->terminal) return r;

    if (a == ACT_UP && env->move_timer == 0) {
        env->move_timer = PLAYER_SPEED;
        env->pos = (env->pos - 1 < 0) ? 0 : env->pos - 1;
    } else if (a == ACT_DOWN && env->move_timer == 0) {
        env->move_timer = PLAYER_SPEED;
        env->pos = (env->pos + 1 > 9) ? 9 : env->pos + 1;
    }

    if (env->pos == 0) {
        r += 1.0f;
        randomize_cars(env, 0);
        env->pos = 9;
    }

    for (int i = 0; i < NUM_CARS; i++) {
        Car* car = &env->cars[i];
        if (car->x == 4 && car->y == env->pos) {
            env->pos = 9;
        }
        if (car->countdown == 0) {
            car->countdown = abs(car->speed);
            car->x += (car->speed > 0) ? 1 : -1;
            if (car->x < 0) car->x = 9;
            else if (car->x > 9) car->x = 0;
            if (car->x == 4 && car->y == env->pos) {
                env->pos = 9;
            }
        } else {
            car->countdown -= 1;
        }
    }

    if (env->move_timer > 0) env->move_timer -= 1;
    env->terminate_timer -= 1;
    if (env->terminate_timer < 0) {
        env->terminal = 1;
    }
    return r;
}

void c_step(MinatarFreeway* env) {
    env->terminals[0] = 0.0f;
    env->rewards[0] = 0.0f;

    int a = (int)env->actions[0];
    if (a < 0) a = 0;
    if (a > 5) a = 5;

    if (rndf(env) < env->sticky_action_prob) a = env->last_action;
    env->last_action = a;

    float r = act(env, a);
    env->rewards[0] = r;
    env->episode_return += r;
    env->episode_length += 1;

    int done = env->terminal;
    if (env->max_steps > 0 && env->episode_length >= env->max_steps) done = 1;

    if (done) {
        env->terminals[0] = 1.0f;
        add_log(env);
        c_reset(env);
    } else {
        compute_observations(env);
    }
}

#define MINATAR_CELL 48
static const Color MINATAR_BG = (Color){6, 24, 24, 255};

static const Color FREEWAY_COLORS[MINATAR_CHANNELS] = {
    (Color){241, 241, 241, 255},
    (Color){187, 0, 0, 255},
    (Color){0, 187, 187, 255},
    (Color){0, 150, 187, 255},
    (Color){0, 110, 187, 255},
    (Color){0, 70, 187, 255},
    (Color){0, 30, 187, 255},
};

void c_render(MinatarFreeway* env) {
    if (!IsWindowReady()) {
        InitWindow(MINATAR_GRID*MINATAR_CELL, MINATAR_GRID*MINATAR_CELL,
            "PufferLib MinAtar Freeway");
        SetTargetFPS(10);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(MINATAR_BG);
    unsigned char* o = env->observations;
    for (int y = 0; y < MINATAR_GRID; y++) {
        for (int x = 0; x < MINATAR_GRID; x++) {
            for (int c = 0; c < MINATAR_CHANNELS; c++) {
                if (o[(y*MINATAR_GRID + x)*MINATAR_CHANNELS + c]) {
                    DrawRectangle(x*MINATAR_CELL, y*MINATAR_CELL,
                        MINATAR_CELL, MINATAR_CELL, FREEWAY_COLORS[c]);
                }
            }
        }
    }
    DrawText(TextFormat("Return: %.0f", env->episode_return), 8, 8, 20, WHITE);
    EndDrawing();
}
