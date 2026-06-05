#include "minatar_asterix.h"

int main() {
    MinatarAsterix env = {
        .sticky_action_prob = 0.1f,
        .ramping = 1,
        .max_steps = 0,
        .num_agents = 1,
        .rng = 0,
    };
    env.observations = (unsigned char*)calloc(MINATAR_OBS_SIZE, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    init(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        int a = ACT_NOOP;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) a = ACT_LEFT;
            else if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) a = ACT_UP;
            else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) a = ACT_RIGHT;
            else if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) a = ACT_DOWN;
        } else {
            a = rand() % 6;
        }
        env.actions[0] = (float)a;
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
