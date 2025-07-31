#pragma once

#include <string_view>
#include <memory>
#include "matrix.hpp"

void load_flag_from_pgm(
    const matrix<char>& flag,
    const int& imax,
    const int& jmax,
    const std::string_view& filename
);

void init_flag(
    const matrix<char>& flag,
    const int& imax,
    const int& jmax,
    const float& delx,
    const float& dely,
    const std::shared_ptr<int>& ibound
);
