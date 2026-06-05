#include "minatar_breakout.h"
#define OBS_SIZE MINATAR_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {6}
#define OBS_TENSOR_T ByteTensor

#define Env MinatarBreakout
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->sticky_action_prob = dict_get(kwargs, "sticky_action_prob")->value;
    env->ramping = (int)dict_get(kwargs, "ramping")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
