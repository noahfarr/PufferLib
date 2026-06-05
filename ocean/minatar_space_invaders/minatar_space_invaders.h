#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define MINATAR_GRID 10
#define MINATAR_CHANNELS 6
#define MINATAR_OBS_SIZE (MINATAR_GRID*MINATAR_GRID*MINATAR_CHANNELS)

#define SHOT_COOL_DOWN 5
#define ENEMY_MOVE_INTERVAL 12
#define ENEMY_SHOT_INTERVAL 10

#define ACT_NOOP  0
#define ACT_LEFT  1
#define ACT_UP    2
#define ACT_RIGHT 3
#define ACT_DOWN  4
#define ACT_FIRE  5

#define CH_CANNON          0
#define CH_ALIEN           1
#define CH_ALIEN_LEFT      2
#define CH_ALIEN_RIGHT     3
#define CH_FRIENDLY_BULLET 4
#define CH_ENEMY_BULLET    5

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

    int pos;
    unsigned char f_bullet_map[MINATAR_GRID*MINATAR_GRID];
    unsigned char e_bullet_map[MINATAR_GRID*MINATAR_GRID];
    unsigned char alien_map[MINATAR_GRID*MINATAR_GRID];
    int alien_dir;
    int enemy_move_interval;
    int alien_move_timer;
    int alien_shot_timer;
    int ramp_index;
    int shot_timer;
    int terminal;

    float episode_return;
    int episode_length;
    int max_steps;

    float sticky_action_prob;
    int ramping;
    int last_action;

    unsigned int rng;
} MinatarSpaceInvaders;

static inline float rndf(MinatarSpaceInvaders* env) {
    return (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
}

void init(MinatarSpaceInvaders* env) {

    (void)env;
}

void c_close(MinatarSpaceInvaders* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

static int count_nonzero(unsigned char* map) {
    int n = 0;
    for (int i = 0; i < MINATAR_GRID*MINATAR_GRID; i++) n += map[i] ? 1 : 0;
    return n;
}

static int any_nonzero(unsigned char* map) {
    for (int i = 0; i < MINATAR_GRID*MINATAR_GRID; i++) if (map[i]) return 1;
    return 0;
}

static int col_sum(unsigned char* map, int x) {
    int n = 0;
    for (int y = 0; y < MINATAR_GRID; y++) n += map[y*MINATAR_GRID + x] ? 1 : 0;
    return n;
}

static int row_sum(unsigned char* map, int y) {
    int n = 0;
    for (int x = 0; x < MINATAR_GRID; x++) n += map[y*MINATAR_GRID + x] ? 1 : 0;
    return n;
}

void compute_observations(MinatarSpaceInvaders* env) {
    memset(env->observations, 0, MINATAR_OBS_SIZE);
    unsigned char* o = env->observations;
    o[(9*MINATAR_GRID + env->pos)*MINATAR_CHANNELS + CH_CANNON] = 1;
    int dir_channel = (env->alien_dir < 0) ? CH_ALIEN_LEFT : CH_ALIEN_RIGHT;
    for (int y = 0; y < MINATAR_GRID; y++) {
        for (int x = 0; x < MINATAR_GRID; x++) {
            int idx = y*MINATAR_GRID + x;
            int base = idx*MINATAR_CHANNELS;
            if (env->alien_map[idx]) {
                o[base + CH_ALIEN] = 1;
                o[base + dir_channel] = 1;
            }
            if (env->f_bullet_map[idx]) o[base + CH_FRIENDLY_BULLET] = 1;
            if (env->e_bullet_map[idx]) o[base + CH_ENEMY_BULLET] = 1;
        }
    }
}

void c_reset(MinatarSpaceInvaders* env) {
    env->pos = 5;
    memset(env->f_bullet_map, 0, sizeof(env->f_bullet_map));
    memset(env->e_bullet_map, 0, sizeof(env->e_bullet_map));
    memset(env->alien_map, 0, sizeof(env->alien_map));

    for (int y = 0; y < 4; y++)
        for (int x = 2; x < 8; x++)
            env->alien_map[y*MINATAR_GRID + x] = 1;
    env->alien_dir = -1;
    env->enemy_move_interval = ENEMY_MOVE_INTERVAL;
    env->alien_move_timer = env->enemy_move_interval;
    env->alien_shot_timer = ENEMY_SHOT_INTERVAL;
    env->ramp_index = 0;
    env->shot_timer = 0;
    env->terminal = 0;

    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->last_action = ACT_NOOP;
    compute_observations(env);
}

void add_log(MinatarSpaceInvaders* env) {
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->episode_return;
    env->log.score += env->episode_return;
    env->log.perf += env->episode_return;
    env->log.n += 1.0f;
}

static int nearest_alien(MinatarSpaceInvaders* env, int pos, int* out_row, int* out_col) {

    for (int d = 0; d < MINATAR_GRID; d++) {
        for (int x = 0; x < MINATAR_GRID; x++) {
            if (abs(x - pos) != d) continue;
            if (col_sum(env->alien_map, x) > 0) {

                for (int y = MINATAR_GRID - 1; y >= 0; y--) {
                    if (env->alien_map[y*MINATAR_GRID + x]) {
                        *out_row = y;
                        *out_col = x;
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

static float act(MinatarSpaceInvaders* env, int a) {
    float r = 0.0f;
    if (env->terminal) return r;

    if (a == ACT_FIRE && env->shot_timer == 0) {
        env->f_bullet_map[9*MINATAR_GRID + env->pos] = 1;
        env->shot_timer = SHOT_COOL_DOWN;
    } else if (a == ACT_LEFT) {
        env->pos = (env->pos - 1 < 0) ? 0 : env->pos - 1;
    } else if (a == ACT_RIGHT) {
        env->pos = (env->pos + 1 > 9) ? 9 : env->pos + 1;
    }

    memmove(env->f_bullet_map, env->f_bullet_map + MINATAR_GRID,
        (MINATAR_GRID*MINATAR_GRID - MINATAR_GRID) * sizeof(unsigned char));
    memset(env->f_bullet_map + (MINATAR_GRID*MINATAR_GRID - MINATAR_GRID), 0, MINATAR_GRID);

    memmove(env->e_bullet_map + MINATAR_GRID, env->e_bullet_map,
        (MINATAR_GRID*MINATAR_GRID - MINATAR_GRID) * sizeof(unsigned char));
    memset(env->e_bullet_map, 0, MINATAR_GRID);
    if (env->e_bullet_map[9*MINATAR_GRID + env->pos]) {
        env->terminal = 1;
    }

    if (env->alien_map[9*MINATAR_GRID + env->pos]) {
        env->terminal = 1;
    }
    if (env->alien_move_timer == 0) {
        int nz = count_nonzero(env->alien_map);
        env->alien_move_timer = (nz < env->enemy_move_interval) ? nz : env->enemy_move_interval;
        if ((col_sum(env->alien_map, 0) > 0 && env->alien_dir < 0) ||
            (col_sum(env->alien_map, 9) > 0 && env->alien_dir > 0)) {
            env->alien_dir = -env->alien_dir;
            if (row_sum(env->alien_map, 9) > 0) {
                env->terminal = 1;
            }

            unsigned char tmp[MINATAR_GRID*MINATAR_GRID];
            for (int y = 0; y < MINATAR_GRID; y++) {
                int src = (y - 1 + MINATAR_GRID) % MINATAR_GRID;
                for (int x = 0; x < MINATAR_GRID; x++)
                    tmp[y*MINATAR_GRID + x] = env->alien_map[src*MINATAR_GRID + x];
            }
            memcpy(env->alien_map, tmp, sizeof(tmp));
        } else {

            unsigned char tmp[MINATAR_GRID*MINATAR_GRID];
            for (int x = 0; x < MINATAR_GRID; x++) {
                int src = (x - env->alien_dir % MINATAR_GRID + MINATAR_GRID) % MINATAR_GRID;
                for (int y = 0; y < MINATAR_GRID; y++)
                    tmp[y*MINATAR_GRID + x] = env->alien_map[y*MINATAR_GRID + src];
            }
            memcpy(env->alien_map, tmp, sizeof(tmp));
        }
        if (env->alien_map[9*MINATAR_GRID + env->pos]) {
            env->terminal = 1;
        }
    }
    if (env->alien_shot_timer == 0) {
        env->alien_shot_timer = ENEMY_SHOT_INTERVAL;
        int nr, nc;
        if (nearest_alien(env, env->pos, &nr, &nc)) {
            env->e_bullet_map[nr*MINATAR_GRID + nc] = 1;
        }
    }

    for (int i = 0; i < MINATAR_GRID*MINATAR_GRID; i++) {
        if (env->alien_map[i] && (env->alien_map[i] == env->f_bullet_map[i])) {
            r += 1.0f;
            env->alien_map[i] = 0;
            env->f_bullet_map[i] = 0;
        }
    }

    if (env->shot_timer > 0) env->shot_timer -= 1;
    env->alien_move_timer -= 1;
    env->alien_shot_timer -= 1;
    if (!any_nonzero(env->alien_map)) {
        if (env->enemy_move_interval > 6 && env->ramping) {
            env->enemy_move_interval -= 1;
            env->ramp_index += 1;
        }
        for (int y = 0; y < 4; y++)
            for (int x = 2; x < 8; x++)
                env->alien_map[y*MINATAR_GRID + x] = 1;
    }
    return r;
}

void c_step(MinatarSpaceInvaders* env) {
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

static const Color SPACE_INVADERS_COLORS[MINATAR_CHANNELS] = {
    (Color){0, 187, 187, 255},
    (Color){241, 241, 241, 255},
    (Color){120, 120, 120, 255},
    (Color){170, 170, 170, 255},
    (Color){0, 187, 0, 255},
    (Color){187, 0, 0, 255},
};

void c_render(MinatarSpaceInvaders* env) {
    if (!IsWindowReady()) {
        InitWindow(MINATAR_GRID*MINATAR_CELL, MINATAR_GRID*MINATAR_CELL,
            "PufferLib MinAtar Space Invaders");
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
                        MINATAR_CELL, MINATAR_CELL, SPACE_INVADERS_COLORS[c]);
                }
            }
        }
    }
    DrawText(TextFormat("Return: %.0f", env->episode_return), 8, 8, 20, WHITE);
    EndDrawing();
}
