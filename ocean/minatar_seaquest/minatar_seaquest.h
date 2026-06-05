#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define MINATAR_GRID 10
#define MINATAR_CHANNELS 10
#define MINATAR_OBS_SIZE (MINATAR_GRID*MINATAR_GRID*MINATAR_CHANNELS)

#define MAX_F_BULLETS 64
#define MAX_E_BULLETS 64
#define MAX_E_FISH    64
#define MAX_E_SUBS    64
#define MAX_DIVERS    64

#define RAMP_INTERVAL       100
#define MAX_OXYGEN          200
#define INIT_SPAWN_SPEED    20
#define DIVER_SPAWN_SPEED   30
#define INIT_MOVE_INTERVAL  5
#define SHOT_COOL_DOWN      5
#define ENEMY_SHOT_INTERVAL 10
#define ENEMY_MOVE_INTERVAL 5
#define DIVER_MOVE_INTERVAL 5

#define ACT_NOOP  0
#define ACT_LEFT  1
#define ACT_UP    2
#define ACT_RIGHT 3
#define ACT_DOWN  4
#define ACT_FIRE  5

#define CH_SUB_FRONT      0
#define CH_SUB_BACK       1
#define CH_FRIENDLY_BULLET 2
#define CH_TRAIL          3
#define CH_ENEMY_BULLET   4
#define CH_ENEMY_FISH     5
#define CH_ENEMY_SUB      6
#define CH_OXYGEN_GUAGE   7
#define CH_DIVER_GUAGE    8
#define CH_DIVER          9

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int active;
    int x, y, dir;
} Bullet;

typedef struct {
    int active;
    int x, y, dir, move_timer;
} Fish;

typedef struct {
    int active;
    int x, y, dir, move_timer, shot_timer;
} Sub;

typedef struct {
    int active;
    int x, y, dir, move_timer;
} Diver;

typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    int oxygen;
    int diver_count;
    int sub_x, sub_y;
    int sub_or;
    int e_spawn_speed;
    int e_spawn_timer;
    int d_spawn_timer;
    int move_speed;
    int ramp_index;
    int shot_timer;
    int surface;
    int terminal;

    Bullet f_bullets[MAX_F_BULLETS];
    Bullet e_bullets[MAX_E_BULLETS];
    Fish   e_fish[MAX_E_FISH];
    Sub    e_subs[MAX_E_SUBS];
    Diver  divers[MAX_DIVERS];

    int n_fbul, n_ebul, n_fish, n_sub, n_div;

    float episode_return;
    int episode_length;
    int max_steps;

    float sticky_action_prob;
    int ramping;
    int last_action;

    unsigned int rng;
} MinatarSeaquest;

static inline float rndf(MinatarSeaquest* env) {
    return (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
}

static inline int rnd_randint(MinatarSeaquest* env, int low, int high) {
    return low + (int)(rand_r(&env->rng) % (unsigned)(high - low));
}

void init(MinatarSeaquest* env) {
    (void)env;
}

void c_close(MinatarSeaquest* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#define COMPACT(arr, n)                            \
    do {                                           \
        int _w = 0;                                \
        for (int _i = 0; _i < (n); _i++) {         \
            if ((arr)[_i].active) {                 \
                if (_w != _i) (arr)[_w] = (arr)[_i];\
                _w++;                              \
            }                                      \
        }                                          \
        (n) = _w;                                  \
    } while (0)

static void compact_all(MinatarSeaquest* env) {
    COMPACT(env->f_bullets, env->n_fbul);
    COMPACT(env->e_bullets, env->n_ebul);
    COMPACT(env->e_fish, env->n_fish);
    COMPACT(env->e_subs, env->n_sub);
    COMPACT(env->divers, env->n_div);
}

void compute_observations(MinatarSeaquest* env) {
    memset(env->observations, 0, MINATAR_OBS_SIZE);
    unsigned char* o = env->observations;
    #define SET(yy, xx, cc) o[((yy)*MINATAR_GRID + (xx))*MINATAR_CHANNELS + (cc)] = 1

    SET(env->sub_y, env->sub_x, CH_SUB_FRONT);
    int back_x = env->sub_or ? env->sub_x - 1 : env->sub_x + 1;
    SET(env->sub_y, back_x, CH_SUB_BACK);

    int ox = env->oxygen;
    if (ox < 0) ox = 0;
    int ox_cols = ox * 10 / MAX_OXYGEN;
    for (int x = 0; x < ox_cols; x++) SET(9, x, CH_OXYGEN_GUAGE);

    for (int x = 9 - env->diver_count; x < 9; x++) SET(9, x, CH_DIVER_GUAGE);

    for (int i = 0; i < env->n_fbul; i++)
        if (env->f_bullets[i].active)
            SET(env->f_bullets[i].y, env->f_bullets[i].x, CH_FRIENDLY_BULLET);

    for (int i = 0; i < env->n_ebul; i++)
        if (env->e_bullets[i].active)
            SET(env->e_bullets[i].y, env->e_bullets[i].x, CH_ENEMY_BULLET);

    for (int i = 0; i < env->n_fish; i++) {
        if (!env->e_fish[i].active) continue;
        Fish* f = &env->e_fish[i];
        SET(f->y, f->x, CH_ENEMY_FISH);
        int bx = f->dir ? f->x - 1 : f->x + 1;
        if (bx >= 0 && bx <= 9) SET(f->y, bx, CH_TRAIL);
    }

    for (int i = 0; i < env->n_sub; i++) {
        if (!env->e_subs[i].active) continue;
        Sub* s = &env->e_subs[i];
        SET(s->y, s->x, CH_ENEMY_SUB);
        int bx = s->dir ? s->x - 1 : s->x + 1;
        if (bx >= 0 && bx <= 9) SET(s->y, bx, CH_TRAIL);
    }

    for (int i = 0; i < env->n_div; i++) {
        if (!env->divers[i].active) continue;
        Diver* d = &env->divers[i];
        SET(d->y, d->x, CH_DIVER);
        int bx = d->dir ? d->x - 1 : d->x + 1;
        if (bx >= 0 && bx <= 9) SET(d->y, bx, CH_TRAIL);
    }
    #undef SET
}

void c_reset(MinatarSeaquest* env) {
    env->oxygen = MAX_OXYGEN;
    env->diver_count = 0;
    env->sub_x = 5;
    env->sub_y = 0;
    env->sub_or = 0;
    memset(env->f_bullets, 0, sizeof(env->f_bullets));
    memset(env->e_bullets, 0, sizeof(env->e_bullets));
    memset(env->e_fish, 0, sizeof(env->e_fish));
    memset(env->e_subs, 0, sizeof(env->e_subs));
    memset(env->divers, 0, sizeof(env->divers));
    env->n_fbul = 0;
    env->n_ebul = 0;
    env->n_fish = 0;
    env->n_sub = 0;
    env->n_div = 0;
    env->e_spawn_speed = INIT_SPAWN_SPEED;
    env->e_spawn_timer = env->e_spawn_speed;
    env->d_spawn_timer = DIVER_SPAWN_SPEED;
    env->move_speed = INIT_MOVE_INTERVAL;
    env->ramp_index = 0;
    env->shot_timer = 0;
    env->surface = 1;
    env->terminal = 0;

    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->last_action = ACT_NOOP;
    compute_observations(env);
}

void add_log(MinatarSeaquest* env) {
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->episode_return;
    env->log.score += env->episode_return;
    env->log.perf += env->episode_return;
    env->log.n += 1.0f;
}

static void add_f_bullet(MinatarSeaquest* env, int x, int y, int dir) {
    if (env->n_fbul < MAX_F_BULLETS)
        env->f_bullets[env->n_fbul++] = (Bullet){1, x, y, dir};
}
static void add_e_bullet(MinatarSeaquest* env, int x, int y, int dir) {
    if (env->n_ebul < MAX_E_BULLETS)
        env->e_bullets[env->n_ebul++] = (Bullet){1, x, y, dir};
}
static void add_e_fish(MinatarSeaquest* env, int x, int y, int dir, int mt) {
    if (env->n_fish < MAX_E_FISH)
        env->e_fish[env->n_fish++] = (Fish){1, x, y, dir, mt};
}
static void add_e_sub(MinatarSeaquest* env, int x, int y, int dir, int mt, int st) {
    if (env->n_sub < MAX_E_SUBS)
        env->e_subs[env->n_sub++] = (Sub){1, x, y, dir, mt, st};
}
static void add_diver(MinatarSeaquest* env, int x, int y, int dir, int mt) {
    if (env->n_div < MAX_DIVERS)
        env->divers[env->n_div++] = (Diver){1, x, y, dir, mt};
}

static void spawn_enemy(MinatarSeaquest* env) {
    int lr = rndf(env) < 0.5f;
    int is_sub = rndf(env) < (1.0f/3.0f);
    int x = lr ? 0 : 9;
    int y = rnd_randint(env, 1, 9);

    for (int i = 0; i < env->n_sub; i++)
        if (env->e_subs[i].active && env->e_subs[i].y == y && env->e_subs[i].dir != lr)
            return;
    for (int i = 0; i < env->n_fish; i++)
        if (env->e_fish[i].active && env->e_fish[i].y == y && env->e_fish[i].dir != lr)
            return;

    if (is_sub) add_e_sub(env, x, y, lr, env->move_speed, ENEMY_SHOT_INTERVAL);
    else        add_e_fish(env, x, y, lr, env->move_speed);
}

static void spawn_diver(MinatarSeaquest* env) {
    int lr = rndf(env) < 0.5f;
    int x = lr ? 0 : 9;
    int y = rnd_randint(env, 1, 9);
    add_diver(env, x, y, lr, DIVER_MOVE_INTERVAL);
}

static float surface_event(MinatarSeaquest* env) {
    env->surface = 1;
    float r;
    if (env->diver_count == 6) {
        env->diver_count = 0;
        r = (float)(env->oxygen * 10 / MAX_OXYGEN);
    } else {
        r = 0.0f;
    }
    env->oxygen = MAX_OXYGEN;
    env->diver_count -= 1;
    if (env->ramping && (env->e_spawn_speed > 1 || env->move_speed > 2)) {
        if (env->move_speed > 2 && (env->ramp_index % 2))
            env->move_speed -= 1;
        if (env->e_spawn_speed > 1)
            env->e_spawn_speed -= 1;
        env->ramp_index += 1;
    }
    return r;
}

static float act(MinatarSeaquest* env, int a) {
    float r = 0.0f;
    if (env->terminal) return r;

    if (env->e_spawn_timer == 0) {
        spawn_enemy(env);
        env->e_spawn_timer = env->e_spawn_speed;
    }
    if (env->d_spawn_timer == 0) {
        spawn_diver(env);
        env->d_spawn_timer = DIVER_SPAWN_SPEED;
    }

    if (a == ACT_FIRE && env->shot_timer == 0) {
        add_f_bullet(env, env->sub_x, env->sub_y, env->sub_or);
        env->shot_timer = SHOT_COOL_DOWN;
    } else if (a == ACT_LEFT) {
        env->sub_x = (env->sub_x - 1 < 0) ? 0 : env->sub_x - 1;
        env->sub_or = 0;
    } else if (a == ACT_RIGHT) {
        env->sub_x = (env->sub_x + 1 > 9) ? 9 : env->sub_x + 1;
        env->sub_or = 1;
    } else if (a == ACT_UP) {
        env->sub_y = (env->sub_y - 1 < 0) ? 0 : env->sub_y - 1;
    } else if (a == ACT_DOWN) {
        env->sub_y = (env->sub_y + 1 > 8) ? 8 : env->sub_y + 1;
    }

    for (int bi = env->n_fbul - 1; bi >= 0; bi--) {
        Bullet* bullet = &env->f_bullets[bi];
        if (!bullet->active) continue;
        bullet->x += bullet->dir ? 1 : -1;
        if (bullet->x < 0 || bullet->x > 9) {
            bullet->active = 0;
        } else {
            int removed = 0;
            for (int xi = 0; xi < env->n_fish; xi++) {
                Fish* x = &env->e_fish[xi];
                if (!x->active) continue;
                if (bullet->x == x->x && bullet->y == x->y) {
                    x->active = 0;
                    bullet->active = 0;
                    r += 1.0f;
                    removed = 1;
                    break;
                }
            }
            if (!removed) {
                for (int xi = 0; xi < env->n_sub; xi++) {
                    Sub* x = &env->e_subs[xi];
                    if (!x->active) continue;
                    if (bullet->x == x->x && bullet->y == x->y) {
                        x->active = 0;
                        bullet->active = 0;
                        r += 1.0f;
                        break;
                    }
                }
            }
        }
    }
    compact_all(env);

    for (int di = env->n_div - 1; di >= 0; di--) {
        Diver* diver = &env->divers[di];
        if (!diver->active) continue;
        if (diver->x == env->sub_x && diver->y == env->sub_y && env->diver_count < 6) {
            diver->active = 0;
            env->diver_count += 1;
        } else {
            if (diver->move_timer == 0) {
                diver->move_timer = DIVER_MOVE_INTERVAL;
                diver->x += diver->dir ? 1 : -1;
                if (diver->x < 0 || diver->x > 9) {
                    diver->active = 0;
                } else if (diver->x == env->sub_x && diver->y == env->sub_y && env->diver_count < 6) {
                    diver->active = 0;
                    env->diver_count += 1;
                }
            } else {
                diver->move_timer -= 1;
            }
        }
    }
    compact_all(env);

    for (int si = env->n_sub - 1; si >= 0; si--) {
        Sub* sub = &env->e_subs[si];
        if (!sub->active) continue;
        if (sub->x == env->sub_x && sub->y == env->sub_y)
            env->terminal = 1;
        if (sub->move_timer == 0) {
            sub->move_timer = env->move_speed;
            sub->x += sub->dir ? 1 : -1;
            if (sub->x < 0 || sub->x > 9) {
                sub->active = 0;
            } else if (sub->x == env->sub_x && sub->y == env->sub_y) {
                env->terminal = 1;
            } else {
                for (int xi = 0; xi < env->n_fbul; xi++) {
                    Bullet* x = &env->f_bullets[xi];
                    if (!x->active) continue;
                    if (sub->x == x->x && sub->y == x->y) {
                        sub->active = 0;
                        x->active = 0;
                        r += 1.0f;
                        break;
                    }
                }
            }
        } else {
            sub->move_timer -= 1;
        }

        if (sub->shot_timer == 0) {
            sub->shot_timer = ENEMY_SHOT_INTERVAL;

            add_e_bullet(env, sub->x, sub->y, sub->dir);
        } else {
            sub->shot_timer -= 1;
        }
    }
    compact_all(env);

    for (int bi = env->n_ebul - 1; bi >= 0; bi--) {
        Bullet* bullet = &env->e_bullets[bi];
        if (!bullet->active) continue;
        if (bullet->x == env->sub_x && bullet->y == env->sub_y)
            env->terminal = 1;
        bullet->x += bullet->dir ? 1 : -1;
        if (bullet->x < 0 || bullet->x > 9) {
            bullet->active = 0;
        } else {
            if (bullet->x == env->sub_x && bullet->y == env->sub_y)
                env->terminal = 1;
        }
    }
    compact_all(env);

    for (int fi = env->n_fish - 1; fi >= 0; fi--) {
        Fish* fish = &env->e_fish[fi];
        if (!fish->active) continue;
        if (fish->x == env->sub_x && fish->y == env->sub_y)
            env->terminal = 1;
        if (fish->move_timer == 0) {
            fish->move_timer = env->move_speed;
            fish->x += fish->dir ? 1 : -1;
            if (fish->x < 0 || fish->x > 9) {
                fish->active = 0;
            } else if (fish->x == env->sub_x && fish->y == env->sub_y) {
                env->terminal = 1;
            } else {
                for (int xi = 0; xi < env->n_fbul; xi++) {
                    Bullet* x = &env->f_bullets[xi];
                    if (!x->active) continue;
                    if (fish->x == x->x && fish->y == x->y) {
                        fish->active = 0;
                        x->active = 0;
                        r += 1.0f;
                        break;
                    }
                }
            }
        } else {
            fish->move_timer -= 1;
        }
    }
    compact_all(env);

    if (env->e_spawn_timer > 0) env->e_spawn_timer -= 1;
    if (env->d_spawn_timer > 0) env->d_spawn_timer -= 1;
    if (env->shot_timer > 0) env->shot_timer -= 1;
    if (env->oxygen <= 0)
        env->terminal = 1;
    if (env->sub_y > 0) {
        env->oxygen -= 1;
        env->surface = 0;
    } else {
        if (!env->surface) {
            if (env->diver_count == 0) {
                env->terminal = 1;
            } else {
                r += surface_event(env);
            }
        }
    }
    return r;
}

void c_step(MinatarSeaquest* env) {
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

static const Color SEAQUEST_COLORS[MINATAR_CHANNELS] = {
    (Color){0, 187, 187, 255},
    (Color){0, 120, 120, 255},
    (Color){241, 241, 241, 255},
    (Color){120, 120, 120, 255},
    (Color){241, 180, 0, 255},
    (Color){187, 0, 0, 255},
    (Color){187, 0, 187, 255},
    (Color){0, 187, 0, 255},
    (Color){0, 0, 187, 255},
    (Color){241, 241, 0, 255},
};

void c_render(MinatarSeaquest* env) {
    if (!IsWindowReady()) {
        InitWindow(MINATAR_GRID*MINATAR_CELL, MINATAR_GRID*MINATAR_CELL,
            "PufferLib MinAtar Seaquest");
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
                        MINATAR_CELL, MINATAR_CELL, SEAQUEST_COLORS[c]);
                }
            }
        }
    }
    DrawText(TextFormat("Return: %.0f", env->episode_return), 8, 8, 20, WHITE);
    EndDrawing();
}
