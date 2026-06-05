#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define MINATAR_GRID 10
#define MINATAR_CHANNELS 4
#define MINATAR_OBS_SIZE (MINATAR_GRID*MINATAR_GRID*MINATAR_CHANNELS)

#define RAMP_INTERVAL     100
#define INIT_SPAWN_SPEED  10
#define INIT_MOVE_INTERVAL 5
#define MINATAR_NUM_ENTITIES 8

#define ACT_NOOP  0
#define ACT_LEFT  1
#define ACT_UP    2
#define ACT_RIGHT 3
#define ACT_DOWN  4
#define ACT_FIRE  5

#define CH_PLAYER 0
#define CH_ENEMY  1
#define CH_TRAIL  2
#define CH_GOLD   3

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int active;
    int x;
    int y;
    int lr;
    int is_gold;
} Entity;

typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    int player_x, player_y;
    Entity entities[MINATAR_NUM_ENTITIES];
    int shot_timer;
    int spawn_speed;
    int spawn_timer;
    int move_speed;
    int move_timer;
    int ramp_timer;
    int ramp_index;
    int terminal;

    float episode_return;
    int episode_length;
    int max_steps;

    float sticky_action_prob;
    int ramping;
    int last_action;

    unsigned int rng;
} MinatarAsterix;

static inline float rndf(MinatarAsterix* env) {
    return (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
}

void init(MinatarAsterix* env) {

    (void)env;
}

void c_close(MinatarAsterix* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

static void spawn_entity(MinatarAsterix* env) {
    int lr = rndf(env) < 0.5f;
    int is_gold = rndf(env) < (1.0f / 3.0f);
    int x = lr ? 0 : 9;

    int slot_options[MINATAR_NUM_ENTITIES];
    int n_options = 0;
    for (int i = 0; i < MINATAR_NUM_ENTITIES; i++) {
        if (!env->entities[i].active) slot_options[n_options++] = i;
    }
    if (n_options == 0) return;

    int slot = slot_options[rand_r(&env->rng) % n_options];
    env->entities[slot].active = 1;
    env->entities[slot].x = x;
    env->entities[slot].y = slot + 1;
    env->entities[slot].lr = lr;
    env->entities[slot].is_gold = is_gold;
}

void compute_observations(MinatarAsterix* env) {
    memset(env->observations, 0, MINATAR_OBS_SIZE);
    unsigned char* o = env->observations;
    o[(env->player_y*MINATAR_GRID + env->player_x)*MINATAR_CHANNELS + CH_PLAYER] = 1;
    for (int i = 0; i < MINATAR_NUM_ENTITIES; i++) {
        Entity* e = &env->entities[i];
        if (!e->active) continue;
        int c = e->is_gold ? CH_GOLD : CH_ENEMY;
        o[(e->y*MINATAR_GRID + e->x)*MINATAR_CHANNELS + c] = 1;
        int back_x = e->lr ? (e->x - 1) : (e->x + 1);
        if (back_x >= 0 && back_x <= 9) {
            o[(e->y*MINATAR_GRID + back_x)*MINATAR_CHANNELS + CH_TRAIL] = 1;
        }
    }
}

void c_reset(MinatarAsterix* env) {
    env->player_x = 5;
    env->player_y = 5;
    for (int i = 0; i < MINATAR_NUM_ENTITIES; i++) env->entities[i].active = 0;
    env->shot_timer = 0;
    env->spawn_speed = INIT_SPAWN_SPEED;
    env->spawn_timer = env->spawn_speed;
    env->move_speed = INIT_MOVE_INTERVAL;
    env->move_timer = env->move_speed;
    env->ramp_timer = RAMP_INTERVAL;
    env->ramp_index = 0;
    env->terminal = 0;

    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->last_action = ACT_NOOP;
    compute_observations(env);
}

void add_log(MinatarAsterix* env) {
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->episode_return;
    env->log.score += env->episode_return;
    env->log.perf += env->episode_return;
    env->log.n += 1.0f;
}

static float act(MinatarAsterix* env, int a) {
    float r = 0.0f;
    if (env->terminal) return r;

    if (env->spawn_timer == 0) {
        spawn_entity(env);
        env->spawn_timer = env->spawn_speed;
    }

    if (a == ACT_LEFT)       env->player_x = (env->player_x - 1 < 0) ? 0 : env->player_x - 1;
    else if (a == ACT_RIGHT) env->player_x = (env->player_x + 1 > 9) ? 9 : env->player_x + 1;
    else if (a == ACT_UP)    env->player_y = (env->player_y - 1 < 1) ? 1 : env->player_y - 1;
    else if (a == ACT_DOWN)  env->player_y = (env->player_y + 1 > 8) ? 8 : env->player_y + 1;

    for (int i = 0; i < MINATAR_NUM_ENTITIES; i++) {
        Entity* e = &env->entities[i];
        if (!e->active) continue;
        if (e->x == env->player_x && e->y == env->player_y) {
            if (e->is_gold) {
                e->active = 0;
                r += 1.0f;
            } else {
                env->terminal = 1;
            }
        }
    }

    if (env->move_timer == 0) {
        env->move_timer = env->move_speed;
        for (int i = 0; i < MINATAR_NUM_ENTITIES; i++) {
            Entity* e = &env->entities[i];
            if (!e->active) continue;
            e->x += e->lr ? 1 : -1;
            if (e->x < 0 || e->x > 9) {
                e->active = 0;
                continue;
            }
            if (e->x == env->player_x && e->y == env->player_y) {
                if (e->is_gold) {
                    e->active = 0;
                    r += 1.0f;
                } else {
                    env->terminal = 1;
                }
            }
        }
    }

    env->spawn_timer -= 1;
    env->move_timer -= 1;

    if (env->ramping && (env->spawn_speed > 1 || env->move_speed > 1)) {
        if (env->ramp_timer >= 0) {
            env->ramp_timer -= 1;
        } else {
            if (env->move_speed > 1 && (env->ramp_index % 2)) {
                env->move_speed -= 1;
            }
            if (env->spawn_speed > 1) {
                env->spawn_speed -= 1;
            }
            env->ramp_index += 1;
            env->ramp_timer = RAMP_INTERVAL;
        }
    }
    return r;
}

void c_step(MinatarAsterix* env) {
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

static const Color ASTERIX_COLORS[MINATAR_CHANNELS] = {
    (Color){0, 187, 187, 255},
    (Color){187, 0, 0, 255},
    (Color){120, 120, 120, 255},
    (Color){241, 200, 0, 255},
};

void c_render(MinatarAsterix* env) {
    if (!IsWindowReady()) {
        InitWindow(MINATAR_GRID*MINATAR_CELL, MINATAR_GRID*MINATAR_CELL,
            "PufferLib MinAtar Asterix");
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
                        MINATAR_CELL, MINATAR_CELL, ASTERIX_COLORS[c]);
                }
            }
        }
    }
    DrawText(TextFormat("Return: %.0f", env->episode_return), 8, 8, 20, WHITE);
    EndDrawing();
}
