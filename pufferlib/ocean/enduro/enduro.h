// puffer_enduro.h

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"

// Constant defs
#define MAX_ENEMIES 10
#define OBSERVATIONS_MAX_SIZE (8 + (4 * MAX_ENEMIES) + 3 + 1)
#define TARGET_FPS 60 // Used to calculate wiggle spawn frequency
#define LOG_BUFFER_SIZE 4096
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 210
#define PLAYABLE_AREA_TOP 0
#define PLAYABLE_AREA_BOTTOM 154
#define PLAYABLE_AREA_LEFT 8
#define PLAYABLE_AREA_RIGHT 160
#define ACTION_HEIGHT (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)
#define CAR_WIDTH 16
#define CAR_HEIGHT 11
#define CRASH_NOOP_DURATION_CAR_VS_CAR 90 // 60 // How long controls are disabled after car v car collision
#define CRASH_NOOP_DURATION_CAR_VS_ROAD 20 // How long controls are disabled after car v road edge collision
#define INITIAL_CARS_TO_PASS 200
#define VANISHING_POINT_X 86 // 110
#define VANISHING_POINT_Y 52
#define VANISHING_POINT_TRANSITION_SPEED 1.0f
#define CURVE_TRANSITION_SPEED 0.05f // 0.02f
#define LOGICAL_VANISHING_Y (VANISHING_POINT_Y + 12)  // Separate logical vanishing point for cars disappearing
#define INITIAL_PLAYER_X 77.0f // 86.0f // ((PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT)/2 + CAR_WIDTH/2)
#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT) // Max y is car length from bottom
#define PLAYER_MIN_Y (ACTION_HEIGHT - CAR_HEIGHT - 9) // Min y is 2 car lengths from bottom
#define ACCELERATION_RATE 0.2f // 0.05f
#define DECELERATION_RATE 0.01f // 0.001f
#define MIN_SPEED -2.5f
#define MAX_SPEED 7.5f // 5.5f
#define ENEMY_CAR_SPEED 0.1f 
// Times of day logic
#define NUM_BACKGROUND_TRANSITIONS 16
// Seconds spent in each time of day
static const float BACKGROUND_TRANSITION_TIMES[] = {
    20.0f, 40.0f, 60.0f, 100.0f, 108.0f, 114.0f, 116.0f, 120.0f,
    124.0f, 130.0f, 134.0f, 138.0f, 170.0f, 198.0f, 214.0f, 232.0f
};

// static const float BACKGROUND_TRANSITION_TIMES[] = {
//     2.0f, 4.0f, 6.0f, 10.0f, 10.8f, 11.0f, 11.6f, 12.0f,
//     12.4f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f
// };

// Curve constants
#define CURVE_STRAIGHT 0
#define CURVE_LEFT -1
#define CURVE_RIGHT 1
#define NUM_LANES 3
// Rendering constants
// Number of digits in the scoreboard
#define SCORE_DIGITS 5
#define CARS_DIGITS  4
// Digit dimensions
#define DIGIT_WIDTH 8
#define DIGIT_HEIGHT 9
// Magic numbers - don't change
#define PLAYER_MIN_X 65.5f
#define PLAYER_MAX_X 91.5f
#define ROAD_LEFT_OFFSET 50.0f
#define ROAD_RIGHT_OFFSET 51.0f
#define VANISHING_POINT_X_LEFT 110.0f
#define VANISHING_POINT_X_RIGHT 62.0f
#define CURVE_VANISHING_POINT_SHIFT 55.0f
#define CURVE_PLAYER_SHIFT_FACTOR 0.025f // Moves player car towards outside edge of curves
// Constants for wiggle effect timing and amplitude
#define WIGGLE_AMPLITUDE 10.0f // 8.0f              // Maximum 'bump-in' offset in pixels
#define WIGGLE_SPEED 10.1f // 10.1f                 // Speed at which the wiggle moves down the screen
#define WIGGLE_LENGTH 26.0f // 26.0f                // Vertical length of the wiggle effect

// Log structs
typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
    float reward;
    float step_rew_car_passed_no_crash;
    float stay_on_road_reward;
    float crashed_penalty;
    float passed_cars;
    float passed_by_enemy;
    int cars_to_pass;
    float days_completed;
    float days_failed;
    float collisions_player_vs_car;
    float collisions_player_vs_road;
} Log;

typedef struct LogBuffer {
    Log* logs;
    int length;
    int idx;
} LogBuffer;

// Car struct for enemy cars
typedef struct Car {
    int lane;   // Lane index: 0=left lane, 1=mid, 2=right lane
    float x;    // Current x position
    float y;    // Current y position
    float last_y; // y post last step
    int passed; // Flag to indicate if car has been passed by player
    int colorIndex; // Car color idx (0-5)
} Car;

// Rendering struct
typedef struct GameState {
    Texture2D backgroundTextures[16]; // 16 different backgrounds for time of day
    Texture2D digitTextures[10];      // Textures for digits 0-9
    Texture2D carDigitTexture;        // Texture for the "CAR" digit
    Texture2D mountainTextures[16];   // Mountain textures corresponding to backgrounds
    Texture2D levelCompleteFlagLeftTexture;  // Texture for left flag
    Texture2D levelCompleteFlagRightTexture; // Texture for right flag
    Texture2D greenDigitTextures[10];        // Textures for green digits
    Texture2D yellowDigitTextures[10];       // Textures for yellow digits
    Texture2D playerCarLeftTreadTexture;
    Texture2D playerCarRightTreadTexture;
    Texture2D enemyCarTextures[6][2]; // [color][tread] 6 colors, 2 treads (left, right)
    Texture2D enemyCarNightTailLightsTexture;
    Texture2D enemyCarNightFogTailLightsTexture;
    // For car animation
    float carAnimationTimer;
    float carAnimationInterval;
    unsigned char showLeftTread;
    float mountainPosition; // Position of the mountain texture
    // Variables for alternating flags
    unsigned char victoryAchieved;
    int flagTimer;
    unsigned char showLeftFlag; // true shows left flag, false shows right flag
    int victoryDisplayTimer;    // Timer for how long victory effects have been displayed
    // Variables for scrolling yellow digits
    float yellowDigitOffset; // Offset for scrolling effect
    int yellowDigitCurrent;  // Current yellow digit being displayed
    int yellowDigitNext;     // Next yellow digit to scroll in
    float scoreIncrementBuffer;
    // Variables for scrolling digits
    float scoreDigitOffsets[SCORE_DIGITS];   // Offset for scrolling effect for each digit
    int scoreDigitCurrents[SCORE_DIGITS];    // Current digit being displayed for each position
    int scoreDigitNexts[SCORE_DIGITS];       // Next digit to scroll in for each position
    unsigned char scoreDigitScrolling[SCORE_DIGITS];  // Scrolling state for each digit
    int scoreTimer; // Timer to control score increment
    int day;
    int carsLeftGameState;
    int score; // Score for scoreboard rendering
    // Background state vars
    float backgroundTransitionTimes[16];
    int backgroundIndex;
    int currentBackgroundIndex;
    int previousBackgroundIndex;
    float elapsedTime;
    // Variable needed from Enduro to maintain separation
    float speed;
    float min_speed;
    float max_speed;
    int current_curve_direction;
    float current_curve_factor;
    float player_x;
    float player_y;
    float initial_player_x;
    float vanishing_point_x;
    float t_p;
    unsigned char dayCompleted;
} GameState;

// Game environment struct
typedef struct Enduro {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncateds;
    LogBuffer* log_buffer;
    Log log;
    size_t obs_size;
    int num_envs;
    float width;
    float height;
    float car_width;
    float car_height;
    int max_enemies;
    float elapsedTimeEnv;
    int initial_cars_to_pass;
    float min_speed;
    float max_speed;
    float player_x;
    float player_y;
    float speed;
    int score;
    int day;
    int lane;
    int step_count;
    int numEnemies;
    int carsToPass;
    float collision_cooldown_car_vs_car; // Timer for car vs car collisions
    float collision_cooldown_car_vs_road; // Timer for car vs road edge collisions
    int drift_direction; // Which way player car drifts whilst noops after crash w/ other car
    float action_height;
    Car enemyCars[MAX_ENEMIES];
    float initial_player_x;
    float road_scroll_offset;
    // Road curve variables
    int current_curve_stage;
    int steps_in_current_stage;
    int current_curve_direction; // 1: Right, -1: Left, 0: Straight
    float current_curve_factor;
    float target_curve_factor;
    float current_step_threshold;
    float target_vanishing_point_x;     // Next curve direction vanishing point
    float current_vanishing_point_x;    // Current interpolated vanishing point
    float base_target_vanishing_point_x; // Target for the base vanishing point
    float vanishing_point_x;
    float base_vanishing_point_x;
    float t_p;
    // Roadside wiggle effect
    float wiggle_y;            // Current y position of the wiggle
    float wiggle_speed;        // Speed at which the wiggle moves down the screen
    float wiggle_length;       // Vertical length of the wiggle effect
    float wiggle_amplitude;    // How far into road wiggle extends
    unsigned char wiggle_active;        // Whether the wiggle is active
    // Player car acceleration
    int currentGear;
    float gearSpeedThresholds[4]; // Speeds at which gear changes occur
    float gearAccelerationRates[4]; // Acceleration rates per gear
    float gearTimings[4]; // Time durations per gear
    float gearElapsedTime; // Time spent in current gear
    // Enemy spawning
    float enemySpawnTimer;
    float enemySpawnInterval; // Spawn interval based on current stage

    // Enemy movement speed
    float enemySpeed;

    // Day completed (victory) variables
    unsigned char dayCompleted;

    // Logging
    float last_road_left;
    float last_road_right;
    int closest_edge_lane;
    int last_spawned_lane;
    float totalAccelerationTime;

    // Rendering
    float parallaxFactor;

    // Variables for time of day
    float dayTransitionTimes[NUM_BACKGROUND_TRANSITIONS];
    int dayTimeIndex;
    int currentDayTimeIndex;
    int previousDayTimeIndex;

    // RNG
    unsigned int rng_state;
    unsigned int index;
    int reset_count;

    // Rewards
    // Reward flag for stepwise rewards if car passed && !crashed
    // Effectively spreads out reward for passing cars
    unsigned char car_passed_no_crash_active;
    float step_rew_car_passed_no_crash;
    float crashed_penalty;
} Enduro;

// Action enumeration
typedef enum {
    ACTION_NOOP = 0,
    ACTION_FIRE = 1,
    ACTION_RIGHT = 2,
    ACTION_LEFT = 3,
    ACTION_DOWN = 4,
    ACTION_DOWNRIGHT = 5,
    ACTION_DOWNLEFT = 6,
    ACTION_RIGHTFIRE = 7,
    ACTION_LEFTFIRE = 8,
} Action;

// Client struct
typedef struct Client {
    float width;
    float height;
    GameState gameState;
} Client;

// Prototypes
// RNG
unsigned int xorshift32(unsigned int *state);
// LogBuffer functions
LogBuffer* allocate_logbuffer(int size);
void free_logbuffer(LogBuffer* buffer);
void add_log(LogBuffer* logs, Log* log);
Log aggregate_and_clear(LogBuffer* logs);

// Environment functions
void allocate(Enduro* env);
void init(Enduro* env, int seed, int env_index);
void free_allocated(Enduro* env);
void reset_round(Enduro* env);
void reset(Enduro* env);
unsigned char check_collision(Enduro* env, Car* car);
int get_player_lane(Enduro* env);
float get_car_scale(float y);
void add_enemy_car(Enduro* env);
void update_time_of_day(Enduro* env);
void accelerate(Enduro* env);
void compute_enemy_car_rewards(Enduro* env);
void c_step(Enduro* env);
void update_road_curve(Enduro* env);
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t);
float road_edge_x(Enduro* env, float y, float offset, unsigned char left);
float car_x_in_lane(Enduro* env, int lane, float y);
void compute_observations(Enduro* env);

// Client functions
Client* make_client(Enduro* env);
void close_client(Client* client, Enduro* env);
void render_car(Client* client, GameState* gameState);
void handleEvents(int* running, Enduro* env);

// GameState rendering functions
void initRaylib();
void loadTextures(GameState* gameState);
void updateCarAnimation(GameState* gameState);
void updateScoreboard(GameState* gameState);
void updateBackground(GameState* gameState);
void renderBackground(GameState* gameState);
void renderScoreboard(GameState* gameState);
void updateMountains(GameState* gameState);
void renderMountains(GameState* gameState);
void updateVictoryEffects(GameState* gameState);
void c_render(Client* client, Enduro* env);
void cleanup(GameState* gameState);

// Function definitions

// RNG
unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// LogBuffer functions
LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
    // printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.score += logs->logs[i].score;
        log.reward += logs->logs[i].reward;
        log.step_rew_car_passed_no_crash += logs->logs[i].step_rew_car_passed_no_crash;
        log.stay_on_road_reward += logs->logs[i].stay_on_road_reward;
        log.crashed_penalty += logs->logs[i].crashed_penalty;
        log.passed_cars += logs->logs[i].passed_cars;
        log.passed_by_enemy += logs->logs[i].passed_by_enemy;
        log.cars_to_pass += logs->logs[i].cars_to_pass;
        log.days_completed += logs->logs[i].days_completed;
        log.days_failed += logs->logs[i].days_failed;
        log.collisions_player_vs_car += logs->logs[i].collisions_player_vs_car;
        log.collisions_player_vs_road += logs->logs[i].collisions_player_vs_road;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    log.reward /= logs->idx;
    log.step_rew_car_passed_no_crash /= logs->idx;
    log.stay_on_road_reward /= logs->idx;
    log.crashed_penalty /= logs->idx;
    log.passed_cars /= logs->idx;
    log.passed_by_enemy /= logs->idx;
    log.cars_to_pass /= logs->idx;
    log.days_completed /= logs->idx;
    log.days_failed /= logs->idx;
    log.collisions_player_vs_car /= logs->idx;
    log.collisions_player_vs_road /= logs->idx;
    logs->idx = 0;
    return log;
}

void init(Enduro* env, int seed, int env_index) {
    env->index = env_index;
    env->rng_state = seed;

    if (seed == 10) { // Activate with seed==0
        // Start the environment at the beginning of the day
        env->elapsedTimeEnv = 0.0f;
        env->currentDayTimeIndex = 0;
        env->previousDayTimeIndex = NUM_BACKGROUND_TRANSITIONS - 1;
    } else {
        // Randomize elapsed time within the day's total duration
        float total_day_duration = BACKGROUND_TRANSITION_TIMES[NUM_BACKGROUND_TRANSITIONS - 1];
        env->elapsedTimeEnv = ((float)xorshift32(&env->rng_state) / (float)UINT32_MAX) * total_day_duration;

        // Determine the current time index
        env->currentDayTimeIndex = 0;
        for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS - 1; i++) {
            if (env->elapsedTimeEnv >= env->dayTransitionTimes[i] &&
                env->elapsedTimeEnv < env->dayTransitionTimes[i + 1]) {
                env->currentDayTimeIndex = i;
                break;
            }
        }

        // Handle the last interval
        if (env->elapsedTimeEnv >= BACKGROUND_TRANSITION_TIMES[NUM_BACKGROUND_TRANSITIONS - 1]) {
            env->currentDayTimeIndex = NUM_BACKGROUND_TRANSITIONS - 1;
        }
    }

    if (env->index % 100 == 0) {
    //printf("Environment #%d state after init: elapsedTimeEnv=%f, currentDayTimeIndex=%d\n",
           //env_index, env->elapsedTimeEnv, env->currentDayTimeIndex);
    }

    env->numEnemies = 0;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].lane = -1; // Default invalid lane
        env->enemyCars[i].y = 0.0f;
        env->enemyCars[i].passed = 0;
    }

    env->obs_size = OBSERVATIONS_MAX_SIZE;
    env->max_enemies = MAX_ENEMIES;
    env->score = 0;
    env->numEnemies = 0;
    env->player_x = INITIAL_PLAYER_X;
    env->player_y = PLAYER_MAX_Y;
    env->speed = MIN_SPEED;
    env->carsToPass = INITIAL_CARS_TO_PASS;
    env->width = SCREEN_WIDTH;
    env->height = SCREEN_HEIGHT;
    env->car_width = CAR_WIDTH;
    env->car_height = CAR_HEIGHT;

    memcpy(env->dayTransitionTimes, BACKGROUND_TRANSITION_TIMES, sizeof(BACKGROUND_TRANSITION_TIMES));
    
    env->step_count = 0;
    env->collision_cooldown_car_vs_car = 0.0f;
    env->collision_cooldown_car_vs_road = 0.0f;
    env->action_height = ACTION_HEIGHT;
    env->elapsedTimeEnv = 0.0f;
    env->enemySpawnTimer = 0.0f;
    env->enemySpawnInterval = 0.8777f;
    env->last_spawned_lane = -1;
    env->closest_edge_lane = -1;
    env->totalAccelerationTime = 0.0f;
    env->base_vanishing_point_x = 86.0f;
    env->current_vanishing_point_x = 86.0f;
    env->target_vanishing_point_x = 86.0f;
    env->vanishing_point_x = 86.0f;
    env->initial_player_x = INITIAL_PLAYER_X;
    env->player_y = PLAYER_MAX_Y;
    env->min_speed = MIN_SPEED;
    env->enemySpeed = ENEMY_CAR_SPEED;
    env->max_speed = MAX_SPEED;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->day = 1;
    env->drift_direction = 0; // Means in noop, but only if crashed state
    env->crashed_penalty = 0.0f;
    env->car_passed_no_crash_active = 1;
    env->step_rew_car_passed_no_crash = 0.0f;
    env->current_curve_stage = 0; // 0: straight
    env->steps_in_current_stage = 0;
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
    env->current_step_threshold = 0.0f;
    env->wiggle_y = VANISHING_POINT_Y;
    env->wiggle_speed = WIGGLE_SPEED;
    env->wiggle_length = WIGGLE_LENGTH;
    env->wiggle_amplitude = WIGGLE_AMPLITUDE;
    env->wiggle_active = true;
    env->currentGear = 0;
    env->gearElapsedTime = 0.0f;
    env->gearTimings[0] = 4.0f;
    env->gearTimings[1] = 2.5f;
    env->gearTimings[2] = 3.25f;
    env->gearTimings[3] = 1.5f;
    float totalSpeedRange = env->max_speed - env->min_speed;
    float totalTime = 0.0f;
    for (int i = 0; i < 4; i++) {
        totalTime += env->gearTimings[i];
    }
    float cumulativeSpeed = env->min_speed;
    for (int i = 0; i < 4; i++) {
        float gearTime = env->gearTimings[i];
        float gearSpeedIncrement = totalSpeedRange * (gearTime / totalTime);
        env->gearSpeedThresholds[i] = cumulativeSpeed + gearSpeedIncrement;
        env->gearAccelerationRates[i] = gearSpeedIncrement / (gearTime * TARGET_FPS);
        cumulativeSpeed = env->gearSpeedThresholds[i];
    }

    // Randomize the initial time of day for each environment
    float total_day_duration = BACKGROUND_TRANSITION_TIMES[15];
    env->elapsedTimeEnv = ((float)rand_r(&env->rng_state) / (float)RAND_MAX) * total_day_duration;
    env->currentDayTimeIndex = 0;
    env->dayTimeIndex = 0;
    env->previousDayTimeIndex = 0;

    // Advance currentDayTimeIndex to match randomized elapsedTimeEnv
    for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS; i++) {
        if (env->elapsedTimeEnv >= env->dayTransitionTimes[i]) {
            env->currentDayTimeIndex = i;
        } else {
            break;
        }
    }

    env->previousDayTimeIndex = (env->currentDayTimeIndex > 0) ? env->currentDayTimeIndex - 1 : NUM_BACKGROUND_TRANSITIONS - 1;
    env->terminals[0] = 0;
    env->truncateds[0] = 0;

    // Reset rewards and logs
    env->rewards[0] = 0.0f;
    env->log.episode_return = 0.0f;
    env->log.episode_length = 0.0f;
    env->log.score = 0.0f;
    env->log.reward = 0.0f;
    env->log.step_rew_car_passed_no_crash = 0.0f;
    env->log.stay_on_road_reward = 0.0f;
    env->log.crashed_penalty = 0.0f;
    env->log.passed_cars = 0.0f;
    env->log.passed_by_enemy = 0.0f;
    env->log.cars_to_pass = INITIAL_CARS_TO_PASS;
    // env->log.days_completed = 0;
    // env->log.days_failed = 0;
    env->log.collisions_player_vs_car = 0.0f;
    env->log.collisions_player_vs_road = 0.0f;
    }

void allocate(Enduro* env) {
    env->rewards = (float*)malloc(sizeof(float) * env->num_envs);
    memset(env->rewards, 0, sizeof(float) * env->num_envs);
    env->observations = (float*)calloc(env->obs_size, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->truncateds = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_allocated(Enduro* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncateds);
    free_logbuffer(env->log_buffer);

}

// Called when a day is failed by player
// Restarts the game at Day 1
void reset_round(Enduro* env) {
    // Preserve RNG state
    unsigned int preserved_rng_state = env->rng_state;
    unsigned int preserved_index = env->index;

    // Reset most environment variables
    env->score = 0;
    env->carsToPass = INITIAL_CARS_TO_PASS;
    env->day = 1;
    env->step_count = 0;
    env->numEnemies = 0;
    env->speed = env->min_speed;
    env->player_x = env->initial_player_x;
    env->player_y = PLAYER_MAX_Y;
    env->car_passed_no_crash_active = 0;
    env->step_rew_car_passed_no_crash = 0.0f;
    env->crashed_penalty = 0.0f;
    env->collision_cooldown_car_vs_car = 0.0f;
    env->collision_cooldown_car_vs_road = 0.0f;
    env->enemySpawnTimer = 0.0f;
    env->enemySpawnInterval = 0.8777f;
    env->last_spawned_lane = -1;
    env->closest_edge_lane = -1;
    env->totalAccelerationTime = 0.0f;
    env->base_vanishing_point_x = 86.0f;
    env->current_vanishing_point_x = 86.0f;
    env->target_vanishing_point_x = 86.0f;
    env->vanishing_point_x = 86.0f;
    env->initial_player_x = INITIAL_PLAYER_X;
    env->player_y = PLAYER_MAX_Y;
    env->min_speed = MIN_SPEED;
    env->enemySpeed = ENEMY_CAR_SPEED;
    env->max_speed = MAX_SPEED;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->day = 1;
    env->drift_direction = 0; // Means in noop, but only if crashed state
    env->crashed_penalty = 0.0f;
    env->car_passed_no_crash_active = 1;
    env->step_rew_car_passed_no_crash = 0.0f;
    env->current_curve_stage = 0; // 0: straight
    env->steps_in_current_stage = 0;
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
    env->current_step_threshold = 0.0f;
    env->wiggle_y = VANISHING_POINT_Y;
    env->wiggle_speed = WIGGLE_SPEED;
    env->wiggle_length = WIGGLE_LENGTH;
    env->wiggle_amplitude = WIGGLE_AMPLITUDE;
    env->wiggle_active = true;
    env->currentGear = 0;
    env->gearElapsedTime = 0.0f;
    env->gearTimings[0] = 4.0f;
    env->gearTimings[1] = 2.5f;
    env->gearTimings[2] = 3.25f;
    env->gearTimings[3] = 1.5f;
    float totalSpeedRange = env->max_speed - env->min_speed;
    float totalTime = 0.0f;
    for (int i = 0; i < 4; i++) {
        totalTime += env->gearTimings[i];
    }
    float cumulativeSpeed = env->min_speed;
    for (int i = 0; i < 4; i++) {
        float gearTime = env->gearTimings[i];
        float gearSpeedIncrement = totalSpeedRange * (gearTime / totalTime);
        env->gearSpeedThresholds[i] = cumulativeSpeed + gearSpeedIncrement;
        env->gearAccelerationRates[i] = gearSpeedIncrement / (gearTime * TARGET_FPS);
        cumulativeSpeed = env->gearSpeedThresholds[i];
    }

    // Reset enemy cars
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].lane = -1;
        env->enemyCars[i].y = 0.0f;
        env->enemyCars[i].passed = 0;
    }

    // Reset rewards and logs
    env->rewards[0] = 0.0f;
    env->log.episode_return = 0.0f;
    env->log.episode_length = 0.0f;
    env->log.score = 0.0f;
    env->log.reward = 0.0f;
    env->log.stay_on_road_reward = 0.0f;
    env->log.crashed_penalty = 0.0f;
    env->log.passed_cars = 0.0f;
    env->log.passed_by_enemy = 0.0f;
    env->log.cars_to_pass = INITIAL_CARS_TO_PASS;
    // env->log.days_completed = 0;
    // env->log.days_failed = 0;
    env->log.collisions_player_vs_car = 0.0f;
    env->log.collisions_player_vs_road = 0.0f;

    // Restore preserved RNG state to maintain reproducibility
    env->rng_state = preserved_rng_state;
    env->index = preserved_index;

    // Restart the environment at the beginning of the day
    env->elapsedTimeEnv = 0.0f;
    env->currentDayTimeIndex = 0;
    env->previousDayTimeIndex = NUM_BACKGROUND_TRANSITIONS - 1;

    // Reset flags and transient states
    env->dayCompleted = 0;
    env->terminals[0] = 0;
    env->truncateds[0] = 0;
    env->rewards[0] = 0.0f;
}

// Reset all init vars
void reset(Enduro* env) {
    // // No random after first reset
    // int reset_seed = (env->reset_count == 0) ? xorshift32(&env->rng_state) : 0;

    int reset_seed = xorshift32(&env->rng_state);
    init(env, reset_seed, env->index);
    env->reset_count += 1;
}

unsigned char check_collision(Enduro* env, Car* car) {
    // // Don't detect multiple collisions in the same step
    // if (env->collision_cooldown_car_vs_car > 0.0f || env->collision_cooldown_car_vs_road > 0.0f) {
    //     return 0;
    // }

    // Compute the scale factor based on vanishing point reference
    float depth = (car->y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth);

    // // Matches render (original game) logic
    // float scale;
    // if (car->y <= 68.0f) scale = 2.0f / 20.0f;
    // else if (car->y <= 74.0f) scale = 4.0f / 20.0f;
    // else if (car->y <= 86.0f) scale = 6.0f / 20.0f;
    // else if (car->y <= 100.0f) scale = 8.0f / 20.0f;
    // else if (car->y <= 110.0f) scale = 12.0f / 20.0f;
    // else if (car->y <= 120.0f) scale = 14.0f / 20.0f;
    // else if (car->y <= 135.0f) scale = 16.0f / 20.0f;
    // else scale = 1.0f;

    float car_width = CAR_WIDTH * scale;
    float car_height = CAR_HEIGHT * scale;
    // Compute car x position
    float car_center_x = car_x_in_lane(env, car->lane, car->y);
    float car_x = car_center_x - car_width / 2.0f;
    return !(env->player_x > car_x + car_width ||
             env->player_x + CAR_WIDTH < car_x ||
             env->player_y > car->y + car_height ||
             env->player_y + CAR_HEIGHT < car->y);
}

// Determines which of the 3 lanes the player's car is in
int get_player_lane(Enduro* env) {
    float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
    float offset = (env->player_x - env->initial_player_x) * 0.5f;
    float left_edge = road_edge_x(env, env->player_y, offset, true);
    float right_edge = road_edge_x(env, env->player_y, offset, false);
    float lane_width = (right_edge - left_edge) / 3.0f;
    env->lane = (int)((player_center_x - left_edge) / lane_width);
    if (env->lane < 0) env->lane = 0;
    if (env->lane > 2) env->lane = 2;
    return env->lane;
}

float get_car_scale(float y) {
    float depth = (y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    return fmaxf(0.1f, 0.9f * depth);
}

void add_enemy_car(Enduro* env) {
    if (env->numEnemies >= MAX_ENEMIES) return;

    int player_lane = get_player_lane(env);
    int furthest_lane;
    int possible_lanes[NUM_LANES];
    int num_possible_lanes = 0;

    // Determine the furthest lane from the player
    if (player_lane == 0) {
        furthest_lane = 2;
    } else if (player_lane == 2) {
        furthest_lane = 0;
    } else {
        // Player is in the middle lane
        // Decide based on player's position relative to the road center
        float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
        float road_center_x = (road_edge_x(env, env->player_y, 0, true) +
                            road_edge_x(env, env->player_y, 0, false)) / 2.0f;
        if (player_center_x < road_center_x) {
            furthest_lane = 2; // Player is on the left side, choose rightmost lane
        } else {
            furthest_lane = 0; // Player is on the right side, choose leftmost lane
        }
    }

    if (env->speed <= 0.0f) {
        // Only spawn in the lane furthest from the player
        possible_lanes[num_possible_lanes++] = furthest_lane;
    } else {
        for (int i = 0; i < NUM_LANES; i++) {
            possible_lanes[num_possible_lanes++] = i;
        }
    }

    if (num_possible_lanes == 0) {
        return; // Rare
    }

    // Randomly select a lane
    int lane = possible_lanes[rand() % num_possible_lanes];
    // Preferentially spawn in the last_spawned_lane 30% of the time
    if (rand() % 100 < 60 && env->last_spawned_lane != -1) {
        lane = env->last_spawned_lane;
    }
    env->last_spawned_lane = lane;
    // Init car
    Car car = { .lane = lane, .x = car_x_in_lane(env, lane, VANISHING_POINT_Y), .y = VANISHING_POINT_Y, .last_y = VANISHING_POINT_Y, .passed = false, .colorIndex = rand() % 6 };
    car.y = (env->speed > 0.0f) ? VANISHING_POINT_Y + 10.0f : PLAYABLE_AREA_BOTTOM + CAR_HEIGHT;
    // Ensure minimum spacing between cars in the same lane
    float depth = (car.y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth + 0.1f);
    float scaled_car_length = CAR_HEIGHT * scale;
    // Randomize min spacing between 1.0f and 6.0f car lengths
    float dynamic_spacing_factor = (rand() / (float)RAND_MAX) * 6.0f + 0.5f;
    float min_spacing = dynamic_spacing_factor * scaled_car_length;
    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        if (existing_car->lane == car.lane) {
            float y_distance = fabs(existing_car->y - car.y);
            if (y_distance < min_spacing) {
                // Too close, do not spawn this car
                return;
            }
        }
    }
    // Ensure not occupying all lanes within vertical range of 6 car lengths
    float min_vertical_range = 6.0f * CAR_HEIGHT;
    int lanes_occupied = 0;
    unsigned char lane_occupied[NUM_LANES] = { false };
    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        float y_distance = fabs(existing_car->y - car.y);
        if (y_distance < min_vertical_range) {
            lane_occupied[existing_car->lane] = true;
        }
    }
    for (int i = 0; i < NUM_LANES; i++) {
        if (lane_occupied[i]) lanes_occupied++;
    }
    if (lanes_occupied >= NUM_LANES - 1 && !lane_occupied[lane]) {
        return;
    }
    env->enemyCars[env->numEnemies++] = car;
}

void update_time_of_day(Enduro* env) {
    float elapsedTime = env->elapsedTimeEnv;
    float totalDuration = env->dayTransitionTimes[15];

    if (elapsedTime >= totalDuration) {
        elapsedTime -= totalDuration;
        env->elapsedTimeEnv = elapsedTime; // Reset elapsed time
        env->dayTimeIndex = 0;
    }

    env->previousDayTimeIndex = env->currentDayTimeIndex;

    while (env->dayTimeIndex < 15 &&
           elapsedTime >= env->dayTransitionTimes[env->dayTimeIndex]) {
        env->dayTimeIndex++;
    }
    env->currentDayTimeIndex = env->dayTimeIndex % 16;
}

void validate_speed(Enduro* env) {
    if (env->speed < env->min_speed || env->speed > env->max_speed) {
        // printf("Speed out of range: %f (min: %f, max: %f)\n", env->speed, env->min_speed, env->max_speed);
        env->speed = fmaxf(env->min_speed, fminf(env->speed, env->max_speed)); // Clamp speed to valid range
    }
}

void validate_gear(Enduro* env) {
    if (env->currentGear < 0 || env->currentGear > 3) {
        // printf("Invalid gear: %d. Resetting to 0.\n", env->currentGear);
        env->currentGear = 0;
    }
}

void accelerate(Enduro* env) {
    validate_speed(env);
    validate_gear(env);

    if (env->speed < env->max_speed) {
        // Gear transition
        if (env->speed >= env->gearSpeedThresholds[env->currentGear] && env->currentGear < 3) {
            env->currentGear++;
            env->gearElapsedTime = 0.0f;
        }

        // Calculate new speed
        float accel = env->gearAccelerationRates[env->currentGear];
        float multiplier = (env->currentGear == 0) ? 4.0f : 2.0f;
        env->speed += accel * multiplier;

        // Clamp speed
        validate_speed(env);

        // Cap speed to gear threshold
        if (env->speed > env->gearSpeedThresholds[env->currentGear]) {
            env->speed = env->gearSpeedThresholds[env->currentGear];
        }
    }
    validate_speed(env);
}

void compute_enemy_car_rewards(Enduro* env) {
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];

        // Calculate the horizontal and vertical distance between the player and the enemy car
        // float car_x = car_x_in_lane(env, car->lane, car->y);
        // float horizontal_distance = fabs(car_x - env->player_x);
        float vertical_distance = fabs(car->y - env->player_y);

        // Manhattan distance
        // float current_distance = sqrt(horizontal_distance * horizontal_distance + vertical_distance * vertical_distance);
        float current_distance = sqrt(vertical_distance * vertical_distance);

        // Compute reward for approaching enemy cars
        if (car->last_y <= env->player_y) {
            float previous_distance = fabs(car->last_y - env->player_y);
            float distance_change = previous_distance - current_distance;

            // printf("Distance change: %.2f\n", distance_change);
            // Reward for getting closer
            if (distance_change > 0 && car->y < env->player_y) { // Enemy car is in front
                //env->rewards[0] += distance_change * 0.0001f; // Scaled reward for reducing distance
                //env->log.reward += distance_change * 0.0001f;
                //printf("Reward for getting closer: %.2f\n", distance_change * 0.0001f);
            }

            // Bonus reward for passing the enemy car
            if (car->y > env->player_y && !car->passed) { // Enemy car is now behind
                env->rewards[0] += 0.5f; // Fixed reward for passing
                env->log.reward += 0.5f;
                car->passed = 1; // Mark car as passed (log-only)
                //printf("Reward for passing enemy car\n");
            }
        }
    }
}

void c_step(Enduro* env) {  
    env->rewards[0] = 0.0f;
    env->elapsedTimeEnv += (1.0f / TARGET_FPS);
    update_time_of_day(env);
    update_road_curve(env);
    env->log.episode_length += 1;
    env->terminals[0] = 0;
    env->road_scroll_offset += env->speed;

    // compute_enemy_car_rewards(env);

    // Update enemy car positions
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        // Compute movement speed adjusted for scaling
        float scale = get_car_scale(car->y);
        float movement_speed = env->speed * scale * 0.75f;
        // Update car position
        car->y += movement_speed;
    }

    // Calculate road edges
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;

    env->last_road_left = road_left;
    env->last_road_right = road_right;

    // Reduced handling on snow
    unsigned char isSnowStage = (env->currentDayTimeIndex == 3);
    float movement_amount = 0.5f; // Default
    if (isSnowStage) {
        movement_amount = 0.3f; // Snow
    }

    // Player movement logic == action space (Discrete[9])
    if (env->collision_cooldown_car_vs_car <= 0 && env->collision_cooldown_car_vs_road <= 0) {
        int act = env->actions[0];
        movement_amount = (((env->speed - env->min_speed) / (env->max_speed - env->min_speed)) + 0.3f); // Magic number
        switch (act) {
            case ACTION_NOOP:
                break;
            case ACTION_FIRE:
                accelerate(env);
                break;
            case ACTION_RIGHT:
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_LEFT:
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
            case ACTION_DOWN:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                break;
            case ACTION_DOWNRIGHT:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_DOWNLEFT:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
            case ACTION_RIGHTFIRE:
                accelerate(env);
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_LEFTFIRE:
                accelerate(env);
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
        
        // Reset crashed_penalty
        env->crashed_penalty = 0.0f;
        }

    } else {

        if (env->collision_cooldown_car_vs_car > 0) {
        env->collision_cooldown_car_vs_car -= 1;
        env->crashed_penalty = -0.01f;
        }
        if (env->collision_cooldown_car_vs_road > 0) {
        env->collision_cooldown_car_vs_road -= 1;
        env->crashed_penalty = -0.01f;
        }

        // Drift towards furthest road edge
        if (env->drift_direction == 0) { // drift_direction is 0 when noop starts
            env->drift_direction = (env->player_x > (road_left + road_right) / 2) ? -1 : 1;
            // Remove enemy cars in middle lane and lane player is drifting towards
            // only if they are behind the player (y > player_y) to avoid crashes
            for (int i = 0; i < env->numEnemies; i++) {
                Car* car = &env->enemyCars[i];

                if ((car->lane == 1 || car->lane == env->lane + env->drift_direction) && (car->y > env->player_y)) {
                    for (int j = i; j < env->numEnemies - 1; j++) {
                        env->enemyCars[j] = env->enemyCars[j + 1];
                    }
                    env->numEnemies--;
                    i--;
                }
            }
        }
        // Drift distance per step
        if (env->collision_cooldown_car_vs_road > 0) {
            env->player_x += env->drift_direction * 0.12f;
        } else {
        env->player_x += env->drift_direction * 0.25f;
        }
    }

    // Road curve/vanishing point movement logic
    // Adjust player's x position based on the current curve
    float curve_shift = -env->current_curve_factor * CURVE_PLAYER_SHIFT_FACTOR * fabs(env->speed);
    env->player_x += curve_shift;
    // Clamp player x position to within road edges
    if (env->player_x < road_left) env->player_x = road_left;
    if (env->player_x > road_right) env->player_x = road_right;
    // Update player's horizontal position ratio, t_p
    float t_p = (env->player_x - PLAYER_MIN_X) / (PLAYER_MAX_X - PLAYER_MIN_X);
    t_p = fmaxf(0.0f, fminf(1.0f, t_p));
    env->t_p = t_p;
    // Base vanishing point based on player's horizontal movement (without curve)
    env->base_vanishing_point_x = VANISHING_POINT_X_LEFT - t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
    // Adjust vanishing point based on current curve
    float curve_vanishing_point_shift = env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;
    env->vanishing_point_x = env->base_vanishing_point_x + curve_vanishing_point_shift;

    // After curve shift        
    // Wiggle logic
    if (env->wiggle_active) {
        float min_wiggle_period = 5.8f; // Slow wiggle period at min speed
        float max_wiggle_period = 0.3f; // Fast wiggle period at max speed
        // Apply non-linear scaling for wiggle period using square root
        float speed_normalized = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
        speed_normalized = fmaxf(0.0f, fminf(1.0f, speed_normalized));  // Clamp between 0 and 1
        // Adjust wiggle period with non-linear scale
        float current_wiggle_period = min_wiggle_period - powf(speed_normalized, 0.25) * (min_wiggle_period - max_wiggle_period);
        // Calculate wiggle speed based on adjusted wiggle period
        env->wiggle_speed = (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y) / (current_wiggle_period * TARGET_FPS);
        // Update wiggle position
        env->wiggle_y += env->wiggle_speed;
        // Reset wiggle when it reaches the bottom
        if (env->wiggle_y > PLAYABLE_AREA_BOTTOM) {
            env->wiggle_y = VANISHING_POINT_Y;
        }
    }

    // Player car moves forward slightly according to speed
    // Update player y position based on speed
    env->player_y = PLAYER_MAX_Y - (env->speed - env->min_speed) / (env->max_speed - env->min_speed) * (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // Clamp player_y to measured range
    if (env->player_y < PLAYER_MIN_Y) env->player_y = PLAYER_MIN_Y;
    if (env->player_y > PLAYER_MAX_Y) env->player_y = PLAYER_MAX_Y;

    // Check for and handle collisions between player and road edges
    if (env->player_x <= road_left || env->player_x >= road_right) {
        env->log.collisions_player_vs_road++;
        env->rewards[0] -= 0.5f;
        env->speed = fmaxf((env->speed - 1.25f), MIN_SPEED);
        env->collision_cooldown_car_vs_road = CRASH_NOOP_DURATION_CAR_VS_ROAD;
        env->drift_direction = 0; // Reset drift direction, has priority over car collisions
        env->player_x = fmaxf(road_left + 1, fminf(road_right - 1, env->player_x));        
    }

    // Enemy car logic
    for (int i = 0; i < env->numEnemies; i++) {    
            Car* car = &env->enemyCars[i];

            // Remove off-screen cars that move below the screen
            if (car->y > PLAYABLE_AREA_BOTTOM + CAR_HEIGHT * 5) {
                // Remove car from array if it moves below the screen
                for (int j = i; j < env->numEnemies - 1; j++) {
                    env->enemyCars[j] = env->enemyCars[j + 1];
                }
                env->numEnemies--;
                i--;
                continue;
            }

            // Remove cars that reach or surpass the logical vanishing point if moving up (player speed negative)
            if (env->speed < 0 && car->y <= LOGICAL_VANISHING_Y) {
                // Remove car from array if it reaches the logical vanishing point if moving down (player speed positive)
                for (int j = i; j < env->numEnemies - 1; j++) {
                    env->enemyCars[j] = env->enemyCars[j + 1];
                }
                env->numEnemies--;
                i--;
                continue;
            }
        
            // If the car is behind the player and speed ≤ 0, move it to the furthest lane
            if (env->speed <= 0 && car->y >= env->player_y + CAR_HEIGHT) {
                // Determine the furthest lane
                int furthest_lane;
                int player_lane = get_player_lane(env);
                if (player_lane == 0) {
                    furthest_lane = 2;
                } else if (player_lane == 2) {
                    furthest_lane = 0;
                } else {
                    // Player is in the middle lane
                    // Decide based on player's position relative to the road center
                    float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
                    float road_center_x = (road_edge_x(env, env->player_y, 0, true) +
                                        road_edge_x(env, env->player_y, 0, false)) / 2.0f;
                    if (player_center_x < road_center_x) {
                        furthest_lane = 2; // Player is on the left side
                    } else {
                        furthest_lane = 0; // Player is on the right side
                    }
                }
                car->lane = furthest_lane;
                continue;
            }

            // Check for passing logic **only if not on collision cooldown**
            if (env->speed > 0 && car->last_y < env->player_y + CAR_HEIGHT && car->y >= env->player_y + CAR_HEIGHT && env->collision_cooldown_car_vs_car <= 0 && env->collision_cooldown_car_vs_road <= 0) {
                if (env->carsToPass > 0) {
                    env->carsToPass -= 1;
                }
                if (!car->passed) {
                    env->log.passed_cars += 1;
                    env->score += 1;
                    env->rewards[0] += 1.0f; // Car passed reward
                    env->car_passed_no_crash_active = 1; // Stepwise rewards activated
                    env->step_rew_car_passed_no_crash += 0.001f; // Stepwise reward
                }
                car->passed = true;
            } else if (env->speed < 0 && car->last_y > env->player_y && car->y <= env->player_y) {
                int maxCarsToPass = (env->day == 1) ? 200 : 300; // Day 1: 200 cars, Day 2+: 300 cars
                if (env->carsToPass == maxCarsToPass) {
                    // Do nothing; log the event
                    env->log.passed_by_enemy += 1.0f;
                } else {
                    env->carsToPass += 1;
                    env->log.passed_by_enemy += 1.0f;
                    env->rewards[0] -= 0.1f;
                }
            }

        //     // Did an enemy pass the player?
        //     printf("Car i: %d, Player y: %.2f, Car y: %.2f, Last y: %.2f, passed? %d\n", i, env->player_y, car->y, car->last_y, car->passed);
        //     printf("Speed: %.2f\n", env->speed);
        //     if (env->speed < 0 && car->last_y > env->player_y  + CAR_HEIGHT && car->y <= env->player_y + CAR_HEIGHT ) {
        //         int maxCarsToPass = (env->day == 1) ? 200 : 300; // Day 1: 200 cars, Day 2+: 300 cars
        //         if (env->carsToPass == maxCarsToPass) {
        //             // Do nothing; log the event
        //             env->log.passed_by_enemy += 1.0f;
        //             printf("Passed by enemy but at max carsToPass so no incrementing it\n");
        //         } else {
        //             printf("carsToPass before, passed by enemy: %d\n", env->carsToPass);
        //             env->carsToPass += 1;
        //             printf("carsToPass after, passed by enemy: %d\n", env->carsToPass);
        //             env->log.passed_by_enemy += 1.0f;
        //             env->rewards[0] -= 0.1f;
        //         }
        // }

        // Preserve last y for passing, obs
        car->last_y = car->y;

        // Check for and handle collisions between player and enemy cars
        if (env->collision_cooldown_car_vs_car <= 0) {
            if (check_collision(env, car)) {
                env->log.collisions_player_vs_car++;
                env->rewards[0] -= 0.5f;
                env->speed = 1 + MIN_SPEED;
                env->collision_cooldown_car_vs_car = CRASH_NOOP_DURATION_CAR_VS_CAR;
                env->drift_direction = 0; // Reset drift direction
                env->car_passed_no_crash_active = 0; // Stepwise rewards deactivated until next car passed
                env->step_rew_car_passed_no_crash = 0.0f; // Reset stepwise reward
            }
        }
    }

    // Calculate enemy spawn interval based on player speed and day
    // Values measured from original gameplay
    float min_spawn_interval = 0.8777f; // Minimum spawn interval
    float max_spawn_interval;
    int dayIndex = env->day - 1;
    float maxSpawnIntervals[] = {0.6667f, 0.3614f, 0.5405f};
    int numMaxSpawnIntervals = sizeof(maxSpawnIntervals) / sizeof(maxSpawnIntervals[0]);

    if (dayIndex < numMaxSpawnIntervals) {
        max_spawn_interval = maxSpawnIntervals[dayIndex];
    } else {
        // For days beyond the observed, decrease max_spawn_interval further
        max_spawn_interval = maxSpawnIntervals[numMaxSpawnIntervals - 1] - (dayIndex - numMaxSpawnIntervals + 1) * 0.05f;
        if (max_spawn_interval < 0.2f) {
            max_spawn_interval = 0.2f; // Do not go below 0.2 seconds
        }
    }

    // Ensure min_spawn_interval is greater than or equal to max_spawn_interval
    if (min_spawn_interval < max_spawn_interval) {
        min_spawn_interval = max_spawn_interval;
    }

    // Calculate speed factor for enemy spawning
    float speed_factor = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
    if (speed_factor < 0.0f) speed_factor = 0.0f;
    if (speed_factor > 1.0f) speed_factor = 1.0f;

    // Interpolate between min and max spawn intervals to scale to player speed
    env->enemySpawnInterval = min_spawn_interval - speed_factor * (min_spawn_interval - max_spawn_interval);

    // Update enemy spawn timer
    env->enemySpawnTimer += (1.0f / TARGET_FPS);
    if (env->enemySpawnTimer >= env->enemySpawnInterval) {
        env->enemySpawnTimer -= env->enemySpawnInterval;
        if (env->numEnemies < MAX_ENEMIES) {
            add_enemy_car(env);
        }
    }

    // Day completed logic
    if (env->carsToPass <= 0 && !env->dayCompleted) {
        env->dayCompleted = true;
    }

    // Handle day transition when background cycles back to 0
    if (env->currentDayTimeIndex == 0 && env->previousDayTimeIndex == 15) {
        // Background cycled back to 0
        if (env->dayCompleted) {
            env->log.days_completed += 1;
            env->day += 1;
            env->rewards[0] += 1.0f;
            env->carsToPass = 300; // Always 300 after the first day
            env->dayCompleted = false;
            add_log(env->log_buffer, &env->log);
            compute_observations(env); // Call compute_observations to log
        
        } else {
            // Player failed to pass required cars, soft-reset environment
            env->log.days_failed += 1.0f;
            env->terminals[0] = true;
            add_log(env->log_buffer, &env->log);
            compute_observations(env); // Call compute_observations before reset to log
            reset_round(env); // Reset round == soft reset
            return;
        }
    }

    // Reward each step after a car is passed until a collision occurs. 
    // Then, no rewards per step until next car is passed.
    if (env->car_passed_no_crash_active) {
        env->rewards[0] += env->step_rew_car_passed_no_crash;
    }

    env->rewards[0] += env->crashed_penalty;
    env->log.crashed_penalty = env->crashed_penalty;
    env->log.step_rew_car_passed_no_crash = env->step_rew_car_passed_no_crash;
    env->log.reward = env->rewards[0];
    env->log.episode_return = env->rewards[0];
    env->step_count++;
    env->score += env->rewards[0];
    env->log.score = env->score;
    int local_cars_to_pass = env->carsToPass;
    env->log.cars_to_pass = (int)local_cars_to_pass;

    compute_observations(env);
}

void compute_observations(Enduro* env) {
    float* obs = env->observations;
    int obs_index = 0;

    // All obs normalized to [0, 1]
    // Bounding box around player
    float player_x_norm = (env->player_x - env->last_road_left) / (env->last_road_right  - env->last_road_left);
    float player_y_norm = (PLAYER_MAX_Y - env->player_y) / (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // float player_width_norm = CAR_WIDTH / (env->last_road_right - env->last_road_left);
    // float player_height_norm = CAR_HEIGHT / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);

    // Player position and speed
    // idx 1-3
    obs[obs_index++] = player_x_norm;
    obs[obs_index++] = player_y_norm;
    obs[obs_index++] = (env->speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);

    // Road edges (separate lines for clarity)
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;

    // Road edges and last road edges
    // idx 4-7
    obs[obs_index++] = (road_left - PLAYABLE_AREA_LEFT) /
                       (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (road_right - PLAYABLE_AREA_LEFT) /
                       (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (env->last_road_left - PLAYABLE_AREA_LEFT) /
                        (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (env->last_road_right - PLAYABLE_AREA_LEFT) /
                        (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);

    // Player lane number
    // idx 8
    obs[obs_index++] = (float)get_player_lane(env) / (NUM_LANES - 1);

    // Enemy cars (numEnemies * 4 values) = 10 * 4 = 40 values
    // idx 9-48
    for (int i = 0; i < env->max_enemies; i++) {
        Car* car = &env->enemyCars[i];

        if (car->y > VANISHING_POINT_Y && car->y < PLAYABLE_AREA_BOTTOM) {
            // Normalize car x position relative to road edges
            float car_x_norm = (car->x - env->last_road_left) / (env->last_road_right - env->last_road_left);
            // Normalize car y position relative to the full road height
            float car_y_norm = (PLAYABLE_AREA_BOTTOM - car->y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
            // Calculate delta_y for relative speed
            float delta_y_norm = (car->last_y - car->y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
            // Determine if the car is in the same lane as the player
            int is_same_lane = (car->lane == env->lane);
            // Add normalized car x position
            obs[obs_index++] = car_x_norm;
            // Add normalized car y position
            obs[obs_index++] = car_y_norm;
            // Add normalized delta y (relative speed)
            obs[obs_index++] = delta_y_norm;
            // Add lane information (binary flag for lane match)
            obs[obs_index++] = (float)is_same_lane;
        } else {
            // Default values for cars that don't exist
            obs[obs_index++] = 0.5f; // Neutral x position
            obs[obs_index++] = 0.5f; // Neutral y position
            obs[obs_index++] = 0.0f; // No movement (delta_y = 0)
            obs[obs_index++] = 0.0f; // Not in the same lane
        }
    }

    // Curve direction
    // idx 49
    obs[obs_index++] = (float)(env->current_curve_direction + 1) / 2.0f;

    // Time of day
    // idx 50
    float total_day_length = env->dayTransitionTimes[15];
    obs[obs_index++] = fmodf(env->elapsedTimeEnv, total_day_length) / total_day_length;

    // Cars to pass
    // idx 51
    obs[obs_index++] = (float)env->carsToPass / env->initial_cars_to_pass;
}

// When to curve road and how to curve it, including dense smooth transitions
// An ugly, dense function, but it is necessary
void update_road_curve(Enduro* env) {
    int* current_curve_stage = &env->current_curve_stage;
    int* steps_in_current_stage = &env->steps_in_current_stage;
    
    // Map speed to the scale between 0.5 and 3.5
    float speed_scale = 0.5f + ((fabs(env->speed) / env->max_speed) * (MAX_SPEED - MIN_SPEED));
    float vanishing_point_transition_speed = VANISHING_POINT_TRANSITION_SPEED + speed_scale; 

    // Randomize step thresholds and curve directions
    int step_thresholds[3];
    int curve_directions[3];
    int last_direction = 0; // Tracks the last curve direction, initialized to straight (0)

    for (int i = 0; i < 3; i++) {
        // Generate random step thresholds
        step_thresholds[i] = 1500 + rand() % 3801; // Random value between 1500 and 3800

        // Generate a random curve direction (-1, 0, 1) with rules
        int direction_choices[] = {-1, 0, 1};
        int next_direction;

        do {
            next_direction = direction_choices[rand() % 3];
        } while ((last_direction == -1 && next_direction == 1) || (last_direction == 1 && next_direction == -1));

        curve_directions[i] = next_direction;
        last_direction = next_direction;
    }

    // Use step thresholds and curve directions dynamically
    env->current_step_threshold = step_thresholds[*current_curve_stage % 3];
    (*steps_in_current_stage)++;

    if (*steps_in_current_stage >= step_thresholds[*current_curve_stage % 3]) {
        env->target_curve_factor = (float)curve_directions[*current_curve_stage % 3];
        *steps_in_current_stage = 0;
        *current_curve_stage = (*current_curve_stage + 1) % 3;
    }


    // Determine sizes of step_thresholds and curve_directions
    size_t step_thresholds_size = sizeof(step_thresholds) / sizeof(step_thresholds[0]);
    size_t curve_directions_size = sizeof(curve_directions) / sizeof(curve_directions[0]);

    // Find the maximum size
    size_t max_size = (step_thresholds_size > curve_directions_size) ? step_thresholds_size : curve_directions_size;

    // Adjust arrays dynamically
    int adjusted_step_thresholds[max_size];
    int adjusted_curve_directions[max_size];

    for (size_t i = 0; i < max_size; i++) {
        adjusted_step_thresholds[i] = step_thresholds[i % step_thresholds_size];
        adjusted_curve_directions[i] = curve_directions[i % curve_directions_size];
    }

    // Use adjusted arrays for current calculations
    env->current_step_threshold = adjusted_step_thresholds[*current_curve_stage % max_size];
    (*steps_in_current_stage)++;

    if (*steps_in_current_stage >= adjusted_step_thresholds[*current_curve_stage]) {
        env->target_curve_factor = (float)adjusted_curve_directions[*current_curve_stage % max_size];
        *steps_in_current_stage = 0;
        *current_curve_stage = (*current_curve_stage + 1) % max_size;
    }

    if (env->current_curve_factor < env->target_curve_factor) {
        env->current_curve_factor = fminf(env->current_curve_factor + CURVE_TRANSITION_SPEED, env->target_curve_factor);
    } else if (env->current_curve_factor > env->target_curve_factor) {
        env->current_curve_factor = fmaxf(env->current_curve_factor - CURVE_TRANSITION_SPEED, env->target_curve_factor);
    }
    env->current_curve_direction = fabsf(env->current_curve_factor) < 0.1f ? CURVE_STRAIGHT 
                             : (env->current_curve_factor > 0) ? CURVE_RIGHT : CURVE_LEFT;

    // Move the vanishing point gradually
    env->base_target_vanishing_point_x = VANISHING_POINT_X_LEFT - env->t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
    float target_shift = env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;
    env->target_vanishing_point_x = env->base_target_vanishing_point_x + target_shift;

    if (env->current_vanishing_point_x < env->target_vanishing_point_x) {
        env->current_vanishing_point_x = fminf(env->current_vanishing_point_x + vanishing_point_transition_speed, env->target_vanishing_point_x);
    } else if (env->current_vanishing_point_x > env->target_vanishing_point_x) {
        env->current_vanishing_point_x = fmaxf(env->current_vanishing_point_x - vanishing_point_transition_speed, env->target_vanishing_point_x);
    }
    env->vanishing_point_x = env->current_vanishing_point_x;
}


// B(t) = (1−t)^2 * P0​+2(1−t) * t * P1​+t^2 * P2​, t∈[0,1]
// Quadratic bezier curve helper function
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t) {
    float one_minus_t = 1.0f - t;
    return one_minus_t * one_minus_t * bottom_x + 
           2.0f * one_minus_t * t * control_x + 
           t * t * top_x;
}

// Computes the edges of the road. Use for both L and R. 
// Lots of magic numbers to replicate as exactly as possible
// original Atari 2600 Enduro road rendering.
float road_edge_x(Enduro* env, float y, float offset, unsigned char left) {
    float t = (PLAYABLE_AREA_BOTTOM - y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float base_offset = left ? -ROAD_LEFT_OFFSET : ROAD_RIGHT_OFFSET;
    float bottom_x = env->base_vanishing_point_x + base_offset + offset;
    float top_x = env->current_vanishing_point_x + offset;    
    float edge_x;
    if (fabsf(env->current_curve_factor) < 0.01f) {
        // Straight road
        edge_x = bottom_x + t * (top_x - bottom_x);
    } else {
        // Adjust curve offset based on curve direction
        float curve_offset = (env->current_curve_factor > 0 ? -30.0f : 30.0f) * fabsf(env->current_curve_factor);
        float control_x = bottom_x + (top_x - bottom_x) * 0.5f + curve_offset;
        // Calculate edge using Bézier curve for proper curvature
        edge_x = quadratic_bezier(bottom_x, control_x, top_x, t);
    }

    // Wiggle effect
    float wiggle_offset = 0.0f;
    if (env->wiggle_active && y >= env->wiggle_y && y <= env->wiggle_y + env->wiggle_length) {
        float t_wiggle = (y - env->wiggle_y) / env->wiggle_length; // Ranges from 0 to 1
        // Trapezoidal wave calculation
        if (t_wiggle < 0.15f) {
            // Connection to road edge
            wiggle_offset = env->wiggle_amplitude * (t_wiggle / 0.15f);
        } else if (t_wiggle < 0.87f) {
            // Flat top of wiggle
            wiggle_offset = env->wiggle_amplitude;
        } else {
            // Reconnection to road edge
            wiggle_offset = env->wiggle_amplitude * ((1.0f - t_wiggle) / 0.13f);
        }
        // Wiggle towards road center
        wiggle_offset *= (left ? 1.0f : -1.0f);
        // Scale wiggle offset based on y position, starting at 0.03f at the vanishing point
        float depth = (y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
        float scale = 0.03f + (depth * depth); 
        if (scale > 0.3f) {
            scale = 0.3f;
        }
        wiggle_offset *= scale;
    }
    // Apply the wiggle offset
    edge_x += wiggle_offset;
    return edge_x;
}

// Computes x position of car in a given lane
float car_x_in_lane(Enduro* env, int lane, float y) {
    // Set offset to 0 to ensure enemy cars align with the road rendering
    float offset = 0.0f;
    float left_edge = road_edge_x(env, y, offset, true);
    float right_edge = road_edge_x(env, y, offset, false);
    float lane_width = (right_edge - left_edge) / NUM_LANES;
    return left_edge + lane_width * (lane + 0.5f);
}

// Handles rendering logic
Client* make_client(Enduro* env) {
    Client* client = (Client*)malloc(sizeof(Client));

    // State data from env (Enduro*)
    client->width = env->width;
    client->height = env->height;
    
    initRaylib();
    loadTextures(&client->gameState);

    return client;
}

void close_client(Client* client, Enduro* env) {
    if (client != NULL) {
        cleanup(&client->gameState);
        CloseWindow();
        free(client);
        client = NULL;
    }
}

void render_car(Client* client, GameState* gameState) {
    Texture2D carTexture = gameState->showLeftTread ? gameState->playerCarLeftTreadTexture : gameState->playerCarRightTreadTexture;
    DrawTexture(carTexture, (int)gameState->player_x, (int)gameState->player_y, WHITE);
}

void handleEvents(int* running, Enduro* env) {
    *env->actions = ACTION_NOOP;
    if (WindowShouldClose()) {
        *running = 0;
    }
    unsigned char left = IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A);
    unsigned char right = IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D);
    unsigned char down = IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S);
    unsigned char fire = IsKeyDown(KEY_SPACE); // Fire key
    if (fire) {
        if (right) {
            *env->actions = ACTION_RIGHTFIRE;
        } else if (left) {
            *env->actions = ACTION_LEFTFIRE;
        } else {
            *env->actions = ACTION_FIRE;
        }
    } else if (down) {
        if (right) {
            *env->actions = ACTION_DOWNRIGHT;
        } else if (left) {
            *env->actions = ACTION_DOWNLEFT;
        } else {
            *env->actions = ACTION_DOWN;
        }
    } else if (right) {
        *env->actions = ACTION_RIGHT;
    } else if (left) {
        *env->actions = ACTION_LEFT;
    } else {
        *env->actions = ACTION_NOOP;
    }
}

void initRaylib() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "puffer_enduro");
    SetTargetFPS(60);
}

void loadTextures(GameState* gameState) {
    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 0.05f; // Init; updated based on speed
    gameState->showLeftTread = true;
    gameState->mountainPosition = 0.0f;

    // Initialize victory effect variables
    gameState->showLeftFlag = true;
    gameState->flagTimer = 0;
    gameState->victoryDisplayTimer = 0;
    gameState->victoryAchieved = false;

    // Initialize scoreboard variables
    gameState->score = 0;
    gameState->scoreTimer = 0;
    gameState->carsLeftGameState = 0;
    gameState->day = 1;

    // Initialize score digits arrays
    for (int i = 0; i < SCORE_DIGITS; i++) {
        gameState->scoreDigitCurrents[i] = 0;
        gameState->scoreDigitNexts[i] = 0;
        gameState->scoreDigitOffsets[i] = 0.0f;
        gameState->scoreDigitScrolling[i] = false;
    }

    // Initialize time-of-day variables
    gameState->elapsedTime = 0.0f;
    gameState->currentBackgroundIndex = 0;
    gameState->backgroundIndex = 0;
    gameState->previousBackgroundIndex = 0;

    // Load background and mountain textures for different times of day per original env
    char backgroundFile[40];
    char mountainFile[40];
    for (int i = 0; i < 16; ++i) {
        snprintf(backgroundFile, sizeof(backgroundFile), "resources/puffer_enduro/%d_bg.png", i);
        gameState->backgroundTextures[i] = LoadTexture(backgroundFile);
        snprintf(mountainFile, sizeof(mountainFile), "resources/puffer_enduro/%d_mtns.png", i);
        gameState->mountainTextures[i] = LoadTexture(mountainFile);
    }
    // Load digit textures 0-9
    char filename[100];
    for (int i = 0; i < 10; i++) {
        snprintf(filename, sizeof(filename), "resources/puffer_enduro/digits_%d.png", i);
        gameState->digitTextures[i] = LoadTexture(filename);
    }
    // Load the "car" digit texture
    gameState->carDigitTexture = LoadTexture("resources/puffer_enduro/digits_car.png");
    // Load level complete flag textures
    gameState->levelCompleteFlagLeftTexture = LoadTexture("resources/puffer_enduro/level_complete_flag_left.png");
    gameState->levelCompleteFlagRightTexture = LoadTexture("resources/puffer_enduro/level_complete_flag_right.png");
    // Load green digits for completed days
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "resources/puffer_enduro/green_digits_%d.png", i);
        gameState->greenDigitTextures[i] = LoadTexture(filename);
    }
    // Load yellow digits for scoreboard numbers
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "resources/puffer_enduro/yellow_digits_%d.png", i);
        gameState->yellowDigitTextures[i] = LoadTexture(filename);
    }
    gameState->playerCarLeftTreadTexture = LoadTexture("resources/puffer_enduro/player_car_left_tread.png");
    gameState->playerCarRightTreadTexture = LoadTexture("resources/puffer_enduro/player_car_right_tread.png");
    // Load enemy car textures for each color and tread variant
    gameState->enemyCarTextures[0][0] = LoadTexture("resources/puffer_enduro/enemy_car_blue_left_tread.png");
    gameState->enemyCarTextures[0][1] = LoadTexture("resources/puffer_enduro/enemy_car_blue_right_tread.png");
    gameState->enemyCarTextures[1][0] = LoadTexture("resources/puffer_enduro/enemy_car_gold_left_tread.png");
    gameState->enemyCarTextures[1][1] = LoadTexture("resources/puffer_enduro/enemy_car_gold_right_tread.png");
    gameState->enemyCarTextures[2][0] = LoadTexture("resources/puffer_enduro/enemy_car_pink_left_tread.png");
    gameState->enemyCarTextures[2][1] = LoadTexture("resources/puffer_enduro/enemy_car_pink_right_tread.png");
    gameState->enemyCarTextures[3][0] = LoadTexture("resources/puffer_enduro/enemy_car_salmon_left_tread.png");
    gameState->enemyCarTextures[3][1] = LoadTexture("resources/puffer_enduro/enemy_car_salmon_right_tread.png");
    gameState->enemyCarTextures[4][0] = LoadTexture("resources/puffer_enduro/enemy_car_teal_left_tread.png");
    gameState->enemyCarTextures[4][1] = LoadTexture("resources/puffer_enduro/enemy_car_teal_right_tread.png");
    gameState->enemyCarTextures[5][0] = LoadTexture("resources/puffer_enduro/enemy_car_yellow_left_tread.png");
    gameState->enemyCarTextures[5][1] = LoadTexture("resources/puffer_enduro/enemy_car_yellow_right_tread.png");
    // Load enemy car night tail lights textures
    gameState->enemyCarNightTailLightsTexture = LoadTexture("resources/puffer_enduro/enemy_car_night_tail_lights.png");
    // Load enemy car night fog tail lights texture
    gameState->enemyCarNightFogTailLightsTexture = LoadTexture("resources/puffer_enduro/enemy_car_night_fog_tail_lights.png");
    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 0.05f; // Initial interval, will be updated based on speed
    gameState->showLeftTread = true;
    gameState->mountainPosition = 0.0f;
}

void cleanup(GameState* gameState) {
    // Unload background and mountain textures
    for (int i = 0; i < 16; ++i) {
        UnloadTexture(gameState->backgroundTextures[i]);
        UnloadTexture(gameState->mountainTextures[i]);
    }
    // Unload digit textures
    for (int i = 0; i < 10; ++i) {
        UnloadTexture(gameState->digitTextures[i]);
        UnloadTexture(gameState->greenDigitTextures[i]);
        UnloadTexture(gameState->yellowDigitTextures[i]);
    }
    // Unload "car" digit and flag textures
    UnloadTexture(gameState->carDigitTexture);
    UnloadTexture(gameState->levelCompleteFlagLeftTexture);
    UnloadTexture(gameState->levelCompleteFlagRightTexture);
    // Unload enemy car textures
    for (int color = 0; color < 6; color++) {
        for (int tread = 0; tread < 2; tread++) {
            UnloadTexture(gameState->enemyCarTextures[color][tread]);
        }
    }
    UnloadTexture(gameState->enemyCarNightTailLightsTexture);
    UnloadTexture(gameState->enemyCarNightFogTailLightsTexture);
    // Unload player car textures
    UnloadTexture(gameState->playerCarLeftTreadTexture);
    UnloadTexture(gameState->playerCarRightTreadTexture);
    CloseWindow();
}

void updateCarAnimation(GameState* gameState) {
    // Update the animation interval based on the player's speed
    // Faster speed means faster alternation
    float minInterval = 0.005f;  // Minimum interval at max speed
    float maxInterval = 0.075f;  // Maximum interval at min speed

    float speedRatio = (gameState->speed - gameState->min_speed) / (gameState->max_speed - gameState->min_speed);
    gameState->carAnimationInterval = maxInterval - (maxInterval - minInterval) * speedRatio;

    // Update the animation timer
    gameState->carAnimationTimer += GetFrameTime(); // Time since last frame

    if (gameState->carAnimationTimer >= gameState->carAnimationInterval) {
        gameState->carAnimationTimer = 0.0f;
        gameState->showLeftTread = !gameState->showLeftTread; // Switch texture
    }
}

void updateScoreboard(GameState* gameState) {
    float normalizedSpeed = fminf(fmaxf(gameState->speed, 1.0f), 2.0f);
    // Determine the frame interval for score increment based on speed
    int frameInterval = (int)(30 / normalizedSpeed);
    gameState->scoreTimer++;

    if (gameState->scoreTimer >= frameInterval) {
        gameState->scoreTimer = 0;
        // Increment the score based on normalized speed
        gameState->score += (int)normalizedSpeed;
        if (gameState->score > 99999) {
            gameState->score = 0;
        }
        // Determine which digits have changed and start scrolling them
        int tempScore = gameState->score;
        for (int i = SCORE_DIGITS - 1; i >= 0; i--) {
            int newDigit = tempScore % 10;
            tempScore /= 10;
            if (newDigit != gameState->scoreDigitCurrents[i]) {
                gameState->scoreDigitNexts[i] = newDigit;
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitScrolling[i] = true;
            }
        }
    }
    // Update scrolling digits
    float scrollSpeed = 0.55f * normalizedSpeed;
    for (int i = 0; i < SCORE_DIGITS; i++) {
        if (gameState->scoreDigitScrolling[i]) {
            gameState->scoreDigitOffsets[i] += scrollSpeed; // Scroll speed
            if (gameState->scoreDigitOffsets[i] >= DIGIT_HEIGHT) {
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitCurrents[i] = gameState->scoreDigitNexts[i];
                gameState->scoreDigitScrolling[i] = false; // Stop scrolling
            }
        }
    }
}

void updateBackground(GameState* gameState) {
    float elapsedTime = gameState->elapsedTime;
    float totalDuration = gameState->backgroundTransitionTimes[15];
    if (elapsedTime >= totalDuration) {
        elapsedTime -= totalDuration;
        gameState->elapsedTime = elapsedTime; // Reset elapsed time in env
        gameState->backgroundIndex = 0;
    }
    gameState->previousBackgroundIndex = gameState->currentBackgroundIndex;
    while (gameState->backgroundIndex < 15 &&
           elapsedTime >= gameState->backgroundTransitionTimes[gameState->backgroundIndex]) {
        gameState->backgroundIndex++;
    }
    gameState->currentBackgroundIndex = gameState->backgroundIndex % 16;
}

void renderBackground(GameState* gameState) {
    Texture2D bgTexture = gameState->backgroundTextures[gameState->currentBackgroundIndex];
    if (bgTexture.id != 0) {
        DrawTexture(bgTexture, 0, 0, WHITE);
    }
}

void renderScoreboard(GameState* gameState) {
    int digitWidth = DIGIT_WIDTH;
    int digitHeight = DIGIT_HEIGHT;
    // Convert bottom-left coordinates to top-left origin
    int scoreStartX = 56 + digitWidth;
    int scoreStartY = 173 - digitHeight;
    int dayX = 56;
    int dayY = 188 - digitHeight;
    int carsX = 72;
    int carsY = 188 - digitHeight;

    // Render score with scrolling effect
    for (int i = 0; i < SCORE_DIGITS; ++i) {
        int digitX = scoreStartX + i * digitWidth;
        Texture2D currentDigitTexture;
        Texture2D nextDigitTexture;

        if (i == SCORE_DIGITS - 1) {
            // Use yellow digits for the last digit
            currentDigitTexture = gameState->yellowDigitTextures[gameState->scoreDigitCurrents[i]];
            nextDigitTexture = gameState->yellowDigitTextures[gameState->scoreDigitNexts[i]];
        } else {
            // Use regular digits
            currentDigitTexture = gameState->digitTextures[gameState->scoreDigitCurrents[i]];
            nextDigitTexture = gameState->digitTextures[gameState->scoreDigitNexts[i]];
        }

        if (gameState->scoreDigitScrolling[i]) {
            // Scrolling effect for this digit
            float offset = gameState->scoreDigitOffsets[i];
            // Render current digit moving up
            Rectangle srcRectCurrent = { 0, 0, digitWidth, digitHeight - (int)offset };
            Rectangle destRectCurrent = { digitX, scoreStartY + (int)offset, digitWidth, digitHeight - (int)offset };
            DrawTextureRec(currentDigitTexture, srcRectCurrent, (Vector2){ destRectCurrent.x, destRectCurrent.y }, WHITE);
            // Render next digit coming up from below
            Rectangle srcRectNext = { 0, digitHeight - (int)offset, digitWidth, (int)offset };
            Rectangle destRectNext = { digitX, scoreStartY, digitWidth, (int)offset };
            DrawTextureRec(nextDigitTexture, srcRectNext, (Vector2){ destRectNext.x, destRectNext.y }, WHITE);
        } else {
            // No scrolling, render the current digit normally
            DrawTexture(currentDigitTexture, digitX, scoreStartY, WHITE);
        }
    }

    // Render day number
    int day = gameState->day % 10;
    int dayTextureIndex = day;
    // Pass dayCompleted condition from Enduro to GameState
    if (gameState->dayCompleted) {
        gameState->victoryAchieved = true;
    }
    if (gameState->victoryAchieved) {
        // Green day digits during victory
        Texture2D greenDigitTexture = gameState->greenDigitTextures[dayTextureIndex];
        DrawTexture(greenDigitTexture, dayX, dayY, WHITE);
    } else {
        // Use normal digits
        Texture2D digitTexture = gameState->digitTextures[dayTextureIndex];
        DrawTexture(digitTexture, dayX, dayY, WHITE);
    }

    // Render "CAR" digit or flags for cars to pass
    if (gameState->victoryAchieved) {
        // Alternate between level_complete_flag_left and level_complete_flag_right
        Texture2D flagTexture = gameState->showLeftFlag ? gameState->levelCompleteFlagLeftTexture : gameState->levelCompleteFlagRightTexture;
        Rectangle destRect = { carsX, carsY, digitWidth * 4, digitHeight };
        DrawTextureEx(flagTexture, (Vector2){ destRect.x, destRect.y }, 0.0f, 1.0f, WHITE);
    } else {
        // Render "CAR" label
        DrawTexture(gameState->carDigitTexture, carsX, carsY, WHITE);
        // Render the remaining digits for cars to pass
        int cars = gameState->carsLeftGameState;
        if (cars < 0) cars = 0; // Ensure cars is not negative
        for (int i = 1; i < CARS_DIGITS; ++i) {
            int divisor = (int)pow(10, CARS_DIGITS - i - 1);
            int digit = (cars / divisor) % 10;
            if (digit < 0 || digit > 9) digit = 0; // Clamp digit to valid range
            int digitX = carsX + i * (digitWidth + 1); // Add spacing between digits
            DrawTexture(gameState->digitTextures[digit], digitX, carsY, WHITE);
        }
    }
}

// Triggers the day completed 'victory' display
// Solely for flapping flag visual effect
void updateVictoryEffects(GameState* gameState) {
    if (gameState->victoryAchieved) {
        gameState->flagTimer++;
        // Modulo triggers flag direction change
        // Flag renders in that direction until next change
        if (gameState->flagTimer % 50 == 0) {
            gameState->showLeftFlag = !gameState->showLeftFlag;
        }
        gameState->victoryDisplayTimer++;
        if (gameState->victoryDisplayTimer >= 10) {
            gameState->victoryDisplayTimer = 0;
        }
    }
}

void updateMountains(GameState* gameState) {
    // Mountain scrolling effect when road is curving
    float baseSpeed = 0.0f;
    float curveStrength = fabsf(gameState->current_curve_factor);
    float speedMultiplier = 1.0f; // Scroll speed
    float scrollSpeed = baseSpeed + curveStrength * speedMultiplier;
    int mountainWidth = gameState->mountainTextures[0].width;
    if (gameState->current_curve_direction == 1) { // Turning left
        gameState->mountainPosition += scrollSpeed;
        if (gameState->mountainPosition >= mountainWidth) {
            gameState->mountainPosition -= mountainWidth;
        }
    } else if (gameState->current_curve_direction == -1) { // Turning right
        gameState->mountainPosition -= scrollSpeed;
        if (gameState->mountainPosition <= -mountainWidth) {
            gameState->mountainPosition += mountainWidth;
        }
    }
}

void renderMountains(GameState* gameState) {
    Texture2D mountainTexture = gameState->mountainTextures[gameState->currentBackgroundIndex];
    if (mountainTexture.id != 0) {
        int mountainWidth = mountainTexture.width;
        int mountainY = 45; // y position per original env
        float playerCenterX = (PLAYER_MIN_X + PLAYER_MAX_X) / 2.0f;
        float playerOffset = gameState->player_x - playerCenterX;
        float parallaxFactor = 0.5f;
        float adjustedOffset = -playerOffset * parallaxFactor;
        float mountainX = -gameState->mountainPosition + adjustedOffset;
        BeginScissorMode(PLAYABLE_AREA_LEFT, 0, SCREEN_WIDTH - PLAYABLE_AREA_LEFT, SCREEN_HEIGHT);
        for (int x = (int)mountainX; x < SCREEN_WIDTH; x += mountainWidth) {
            DrawTexture(mountainTexture, x, mountainY, WHITE);
        }
        for (int x = (int)mountainX - mountainWidth; x > -mountainWidth; x -= mountainWidth) {
            DrawTexture(mountainTexture, x, mountainY, WHITE);
        }
        EndScissorMode();
    }
}

void c_render(Client* client, Enduro* env) {
    GameState* gameState = &client->gameState;

    // Copy env state to gameState
    gameState->speed = env->speed;
    gameState->min_speed = env->min_speed;
    gameState->max_speed = env->max_speed;
    gameState->current_curve_direction = env->current_curve_direction;
    gameState->current_curve_factor = env->current_curve_factor;
    gameState->player_x = env->player_x;
    gameState->player_y = env->player_y;
    gameState->initial_player_x = env->initial_player_x;
    gameState->vanishing_point_x = env->vanishing_point_x;
    gameState->t_p = env->t_p;
    gameState->dayCompleted = env->dayCompleted;
    gameState->currentBackgroundIndex = env->currentDayTimeIndex;
    gameState->carsLeftGameState = env->carsToPass;
    gameState->day = env->day;
    gameState->elapsedTime = env->elapsedTimeEnv;

    BeginDrawing();
    ClearBackground(BLACK);
    BeginBlendMode(BLEND_ALPHA);

    renderBackground(gameState);
    updateCarAnimation(gameState);
    updateMountains(gameState);
    renderMountains(gameState);
    
    int bgIndex = gameState->currentBackgroundIndex;
    unsigned char isNightFogStage = (bgIndex == 13);
    unsigned char isNightStage = (bgIndex == 12 || bgIndex == 13 || bgIndex == 14);

    // During night fog stage, clip rendering to y >= 92
    float clipStartY = isNightFogStage ? 92.0f : VANISHING_POINT_Y;
    float clipHeight = PLAYABLE_AREA_BOTTOM - clipStartY;
    Rectangle clipRect = { PLAYABLE_AREA_LEFT, clipStartY, PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT, clipHeight };
    BeginScissorMode(clipRect.x, clipRect.y, clipRect.width, clipRect.height);

    // Render road edges w/ gl lines for classic look
    // During night fog stage, start from y=92
    float roadStartY = isNightFogStage ? 92.0f : VANISHING_POINT_Y;
    Vector2 previousLeftPoint = {0}, previousRightPoint = {0};
    unsigned char firstPoint = true;

    for (float y = roadStartY; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
        float adjusted_y = (env->speed < 0) ? y : y + fmod(env->road_scroll_offset, 0.75f);
        if (adjusted_y > PLAYABLE_AREA_BOTTOM) continue;

        float left_edge = road_edge_x(env, adjusted_y, 0, true);
        float right_edge = road_edge_x(env, adjusted_y, 0, false);

        // Road is multiple shades of gray based on y position
        Color roadColor;
        if (adjusted_y >= 52 && adjusted_y < 91) {
            roadColor = (Color){74, 74, 74, 255};
        } else if (adjusted_y >= 91 && adjusted_y < 106) {
            roadColor = (Color){111, 111, 111, 255};
        } else if (adjusted_y >= 106 && adjusted_y <= 154) {
            roadColor = (Color){170, 170, 170, 255};
        } else {
            roadColor = WHITE;
        }

        Vector2 currentLeftPoint = {left_edge, adjusted_y};
        Vector2 currentRightPoint = {right_edge, adjusted_y};

        if (!firstPoint) {
            DrawLineV(previousLeftPoint, currentLeftPoint, roadColor);
            DrawLineV(previousRightPoint, currentRightPoint, roadColor);
        }

        previousLeftPoint = currentLeftPoint;
        previousRightPoint = currentRightPoint;
        firstPoint = false;
    }

    // Render enemy cars scaled stages for distance/closeness effect
    unsigned char skipFogCars = isNightFogStage;
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        
        // Don't render cars in fog
        if (skipFogCars && car->y < 92.0f) {
            continue;
        }

        // // Determine the car scale based on the seven-stage progression
        // float car_scale;
        // if (car->y <= 68.0f) car_scale = 2.0f / 20.0f;        // Stage 1
        // else if (car->y <= 74.0f) car_scale = 4.0f / 20.0f;   // Stage 2
        // else if (car->y <= 86.0f) car_scale = 6.0f / 20.0f;   // Stage 3
        // else if (car->y <= 100.0f) car_scale = 8.0f / 20.0f;  // Stage 4
        // else if (car->y <= 110.0f) car_scale = 12.0f / 20.0f;  // Stage 5
        // else if (car->y <= 120.0f) car_scale = 14.0f / 20.0f; // Stage 6
        // else if (car->y <= 135.0f) car_scale = 16.0f / 20.0f; // Stage 7
        // else car_scale = 1.0f;                                // Normal size

        float car_scale = get_car_scale(car->y);

        // Select the correct texture based on the car's color and current tread
        Texture2D carTexture;
        if (isNightStage) {
            carTexture = (bgIndex == 13) ? gameState->enemyCarNightFogTailLightsTexture : gameState->enemyCarNightTailLightsTexture;
        } else {
            int colorIndex = car->colorIndex;
            int treadIndex = gameState->showLeftTread ? 0 : 1;
            carTexture = gameState->enemyCarTextures[colorIndex][treadIndex];
        }
        // Compute car coords
        float car_center_x = car_x_in_lane(env, car->lane, car->y);
        //float car_width = CAR_WIDTH * car_scale;
        //float car_height = CAR_HEIGHT * car_scale;
        float car_x = car_center_x - (carTexture.width * car_scale) / 2.0f;
        float car_y = car->y - (carTexture.height * car_scale) / 2.0f;

        DrawTextureEx(carTexture, (Vector2){car_x, car_y}, 0.0f, car_scale, WHITE);
    }

    updateCarAnimation(gameState);
    render_car(client, gameState); // Unscaled player car

    EndScissorMode();
    EndBlendMode();
    updateVictoryEffects(gameState);

    // Update GameState data from environment data;
    client->gameState.victoryAchieved = env->dayCompleted;

    updateScoreboard(gameState);
    renderScoreboard(gameState);
    EndDrawing();
}
