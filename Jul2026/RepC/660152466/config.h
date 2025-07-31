#pragma once

#include "smooth_life.h"

typedef struct {
    unsigned int max_fps;
    SMConfig sm_conf;
} Config;

int gen_config(Config* conf, int argc, const char** argv);
