#ifndef ZFP_MONTECARLO_SIMULATION_HPP
#define ZFP_MONTECARLO_SIMULATION_HPP
#pragma once

#include <vector>

// This function calculates an approximation of the value of pi using
// the Monte Carlo simulation method. The function generates a number
// of random points within a square and counts the number of points
// that fall within a quarter circle inscribed in the square. The ratio
// of the number of points in the quarter circle to the total number of
// points is used to estimate the value of pi.
template <typename T>
T monteCarloPi(int num_samples);

#endif // ZFP_MONTECARLO_SIMULATION_HPP
