#pragma once

#include "matrix.hpp"

void apply_boundary_conditions(
    const matrix<float>& u,
    const matrix<float>& v,
    const matrix<char>& flag,
    const int& imax,
    const int& jmax,
    const float& ui,
    const float& vi
);
