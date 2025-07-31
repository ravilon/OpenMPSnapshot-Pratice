#pragma once
#include "pass_bits/problem.hpp"

namespace pass
{
/**
 * Depending on your problem it returns if you should use openMP or not
 * Outputs are given through all the analyse process.
 *
 * Steps:
 * 1. Estimate the evaluation time of your problem
 * 2. Generate trainingsdata
 * 3. Test linear model if it fits
 * 4. Test polynomial model if it fitst
 * 5. Predict the speedup for your model
 * 6. Give suggestions if to activate openMP or not
 *
 */
bool enable_openmp(const pass::problem &problem);

/**
 * Machine Learning - Step 1.
 *
 * Collects trainingdata for the model.
 * Default: 120 trainingspoint are collected
 * The more data you collect (better data), the better the model performs.
 */
arma::mat train(const int &examples);

/**
 * Machine Learning - Step 2.
 *
 * Builds the model from the trainingdata.
 * Tests two models: linear and polynomial.
 */
arma::rowvec build_model(const arma::mat &training_points);

} // namespace pass
