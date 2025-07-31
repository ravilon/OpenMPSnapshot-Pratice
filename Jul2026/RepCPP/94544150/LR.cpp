//
//  LR.cpp
//  LR-x
//
//  Created by zhuangqh on 2017/6/16.
//  Copyright © 2017年 zhuangqh. All rights reserved.
//

#include <iostream>
#include <vector>
#include <omp.h>
#include <numeric>
#include "LR.hpp"

namespace LR {

  double g(double x) {
    return 1.0 / (1.0 + std::exp(-x));
  }

  VectorXd LogisticRegression::predict(MatrixXd test) {
    return (test * theta).unaryExpr(std::ptr_fun(g));
  }

  void LogisticRegression::regularize(VectorXd &gw) {
    gw += this->lambda / this->m * this->theta;
    gw[0] -= this->lambda / this->m * this->theta(0);
  }

  // Vectorization
  void LogisticRegression::fit_vec(std::pair<Eigen::MatrixXd, Eigen::VectorXd> train) {
    m = train.first.rows();
    n = train.first.cols();

    theta.setZero(n);

    for (size_t i = 0; i < maxIter; i++) {
      // A = x * theta
      VectorXd inner = train.first * theta;

      // E = g(A)
      VectorXd ga = inner.unaryExpr(std::ptr_fun(g));

      // gw = 1/m * xT * (E - y)
      VectorXd gw = 1.0 / this->m * (train.first.transpose() * (ga - train.second));

      regularize(gw);

      // theta -= alpha * gw
      this->theta -= this->alpha * gw;

      // log
      if (verbose) {
        std::cout << "gradient " << gw.norm() << std::endl;
      }

      // check for convergence
      if (this->epsilon > gw.norm()) {
        break;
      }
    }
  }

  void LogisticRegression::fit_naive(std::pair<MatrixXd, VectorXd> train) {
    m = train.first.rows();
    n = train.first.cols();

    theta.setZero(n);

    for (size_t iter = 0; iter < maxIter; iter++) {

      VectorXd gw;
      gw.setZero(n);
      for (size_t i = 0; i < m; i++) {
        double coeff = train.first.row(i) * theta;

        // h(xi) = 1/m * (g(xi * theta) - yi)
        coeff = 1.0 / m * (g(coeff) - train.second(i));

        // gwi = h(xi) * xi
        VectorXd gwi = coeff * train.first.row(i);

        gw += gwi;
      }

      regularize(gw);

      // theta -= alpha * gw
      this->theta -= this->alpha * gw;

      // log
      if (verbose) {
        std::cout << "gradient " << gw.norm() << std::endl;
      }

      // check for convergence
      if (this->epsilon > gw.norm()) {
        break;
      }
    }

  }

  void LogisticRegression::fit_parallel(std::pair<MatrixXd, VectorXd> train) {
    m = train.first.rows();
    n = train.first.cols();

    theta.setZero(n);

    int coreNum = omp_get_num_procs();

    // temp sum in each processor
    std::vector<VectorXd> sumInCore(coreNum);

    // iterator in each processor
    std::vector<size_t> iters(coreNum);

    // iter count for each processor
    size_t sectionNum = m / coreNum;

    for (size_t iter = 0; iter < maxIter; iter++) {

      for (auto &item : sumInCore) {
        item.setZero(n);
      }

      VectorXd gw;
      gw.setZero(n);

#pragma omp parallel for
      for (size_t core = 0; core < coreNum; core++) {

        for (iters[core] = core * sectionNum; iters[core] < (core + 1) * sectionNum && iters[core] < m; iters[core]++) {

            double coeff = train.first.row(iters[core]) * theta;

            // h(xi) = 1/m * (g(xi * theta) - yi)
            coeff = 1.0 / m * (g(coeff) - train.second(iters[core]));

            // gwi = h(xi) * xi
            sumInCore[core] += coeff * train.first.row(iters[core]);

        }

      }

      // merge
      gw = std::accumulate(sumInCore.begin(), sumInCore.end(), gw);

      regularize(gw);

      // theta -= alpha * gw
      this->theta -= this->alpha * gw;

      // log
      if (verbose) {
        std::cout << "gradient " << gw.norm() << std::endl;
      }

      // check for convergence
      if (this->epsilon > gw.norm()) {
        break;
      }
    }

  }

}
