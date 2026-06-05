#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define MINATAR_GRID 10
#define MINATAR_CHANNELS 4
#define MINATAR_OBS_SIZE (MINATAR_GRID*MINATAR_GRID*MINATAR_CHANNELS)

#define ACT_NOOP  0
#define ACT_LEFT  1
#define ACT_UP    2
#define ACT_RIGHT 3
#define ACT_DOWN  4
#define ACT_FIRE  5

#define CH_PADDLE 0
#define CH_BALL   1
#define CH_TRAIL  2
#define CH_BRICK  3

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    int ball_y, ball_x, ball_dir;
    int pos;
    int last_x, last_y;
    int strike;
    unsigned char brick_map[MINATAR_GRID*MINATAR_GRID];
    int terminal;

    float episode_return;
    int episode_length;
    int max_steps;

    float sticky_action_prob;
    int ramping;
    int last_action;

    unsigned int rng;
} MinatarBreakout;

static inline float rndf(MinatarBreakout* env) {
    return (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
}

void init(MinatarBreakout* env) {

    (void)env;
}

void c_close(MinatarBreakout* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

static int count_bricks(MinatarBreakout* env) {
    int n = 0;
    for (int i = 0; i < MINATAR_GRID*MINATAR_GRID; i++) n += env->brick_map[i];
    return n;
}

void compute_observations(MinatarBreakout* env) {
    memset(env->observations, 0, MINATAR_OBS_SIZE);
    unsigned char* o = env->observations;
    o[(env->ball_y*MINATAR_GRID + env->ball_x)*MINATAR_CHANNELS + CH_BALL] = 1;
    o[(9*MINATAR_GRID + env->pos)*MINATAR_CHANNELS + CH_PADDLE] = 1;
    o[(env->last_y*MINATAR_GRID + env->last_x)*MINATAR_CHANNELS + CH_TRAIL] = 1;
    for (int y = 0; y < MINATAR_GRID; y++) {
        for (int x = 0; x < MINATAR_GRID; x++) {
            if (env->brick_map[y*MINATAR_GRID + x])
                o[(y*MINATAR_GRID + x)*MINATAR_CHANNELS + CH_BRICK] = 1;
        }
    }
}

void c_reset(MinatarBreakout* env) {
    env->ball_y = 3;
    int ball_start = rand_r(&env->rng) % 2;
    if (ball_start == 0) { env->ball_x = 0; env->ball_dir = 2; }
    else                 { env->ball_x = 9; env->ball_dir = 3; }
    env->pos = 4;
    memset(env->brick_map, 0, sizeof(env->brick_map));
    for (int y = 1; y < 4; y++)
        for (int x = 0; x < MINATAR_GRID; x++)
            env->brick_map[y*MINATAR_GRID + x] = 1;
    env->strike = 0;
    env->last_x = env->ball_x;
    env->last_y = env->ball_y;
    env->terminal = 0;

    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->last_action = ACT_NOOP;
    compute_observations(env);
}

void add_log(MinatarBreakout* env) {
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->episode_return;
    env->log.score += env->episode_return;
    env->log.perf += env->episode_return;
    env->log.n += 1.0f;
}

static float act(MinatarBreakout* env, int a) {
    float r = 0.0f;
    if (env->terminal) return r;

    if (a == ACT_LEFT)       env->pos = (env->pos - 1 < 0) ? 0 : env->pos - 1;
    else if (a == ACT_RIGHT) env->pos = (env->pos + 1 > 9) ? 9 : env->pos + 1;

    env->last_x = env->ball_x;
    env->last_y = env->ball_y;
    int new_x = env->ball_x, new_y = env->ball_y;
    if (env->ball_dir == 0)      { new_x = env->ball_x - 1; new_y = env->ball_y - 1; }
    else if (env->ball_dir == 1) { new_x = env->ball_x + 1; new_y = env->ball_y - 1; }
    else if (env->ball_dir == 2) { new_x = env->ball_x + 1; new_y = env->ball_y + 1; }
    else if (env->ball_dir == 3) { new_x = env->ball_x - 1; new_y = env->ball_y + 1; }

    static const int reflect_x[4] = {1, 0, 3, 2};
    static const int reflect_y[4] = {3, 2, 1, 0};
    static const int reflect_paddle[4] = {2, 3, 0, 1};

    int strike_toggle = 0;
    if (new_x < 0 || new_x > 9) {
        if (new_x < 0) new_x = 0;
        if (new_x > 9) new_x = 9;
        env->ball_dir = reflect_x[env->ball_dir];
    }
    if (new_y < 0) {
        new_y = 0;
        env->ball_dir = reflect_y[env->ball_dir];
    } else if (env->brick_map[new_y*MINATAR_GRID + new_x] == 1) {
        strike_toggle = 1;
        if (!env->strike) {
            r += 1.0f;
            env->strike = 1;
            env->brick_map[new_y*MINATAR_GRID + new_x] = 0;
            new_y = env->last_y;
            env->ball_dir = reflect_y[env->ball_dir];
        }
    } else if (new_y == 9) {
        if (count_bricks(env) == 0) {
            for (int y = 1; y < 4; y++)
                for (int x = 0; x < MINATAR_GRID; x++)
                    env->brick_map[y*MINATAR_GRID + x] = 1;
        }
        if (env->ball_x == env->pos) {
            env->ball_dir = reflect_y[env->ball_dir];
            new_y = env->last_y;
        } else if (new_x == env->pos) {
            env->ball_dir = reflect_paddle[env->ball_dir];
            new_y = env->last_y;
        } else {
            env->terminal = 1;
        }
    }

    if (!strike_toggle) env->strike = 0;

    env->ball_x = new_x;
    env->ball_y = new_y;
    return r;
}

void c_step(MinatarBreakout* env) {
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

static const Color BREAKOUT_COLORS[MINATAR_CHANNELS] = {
    (Color){0, 187, 187, 255},
    (Color){241, 241, 241, 255},
    (Color){120, 120, 120, 255},
    (Color){187, 0, 0, 255},
};

void c_render(MinatarBreakout* env) {
    if (!IsWindowReady()) {
        InitWindow(MINATAR_GRID*MINATAR_CELL, MINATAR_GRID*MINATAR_CELL,
            "PufferLib MinAtar Breakout");
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
                        MINATAR_CELL, MINATAR_CELL, BREAKOUT_COLORS[c]);
                }
            }
        }
    }
    DrawText(TextFormat("Return: %.0f", env->episode_return), 8, 8, 20, WHITE);
    EndDrawing();
}
