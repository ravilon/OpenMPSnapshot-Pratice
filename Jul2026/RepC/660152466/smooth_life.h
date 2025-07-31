#pragma once

#include <stddef.h>


#define SM_DEFAULT_N_THREADS (4)

// Configuration struct and enum
typedef enum {
    SM_SINGLETHREADED,
    SM_THREAD_POOL,
    SM_OMP
} SMExecutionPolicy;

typedef struct {
    SMExecutionPolicy ex_policy;
    size_t n_threads;                       // If n_threads == 0 it is assumed to be SM_DEFAULT_N_THREADS.
    unsigned int width, height;
    float init_percent_x, init_percent_y;
    float ra, ri;                           // If ri is set < 0 it is assumed to be ri = ra / 3.0f.
    float b1, d1, b2, d2, alpha_m, alpha_n;
    float dt;
} SMConfig;

// State
typedef struct SMState SMState;

SMState* sm_init(SMConfig* conf);
void sm_deinit(SMState* state);
void state_step(SMState* s);
float* sm_get_raw_state(SMState* s);