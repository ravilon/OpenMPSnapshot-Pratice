#pragma once

/* escape time algorithm parameters */
constexpr double BAIL = 4.0;
constexpr int MAX_ITER = 1000;

/* screen size */
constexpr int WIDTH = 1400, HEIGHT = 800;

/* aspect ratio */
constexpr double RATIO = static_cast<double>(WIDTH) / HEIGHT;

/* pan and zoom rates for window */
constexpr double PAN_RATE = 0.1, ZOOM_RATE = 1.1;

