#pragma once

#include "gnu_plot.hpp"

class GameOfLifePlotter : public GNUPlotter {
 public:
  GameOfLifePlotter(int nx, int ny);

  void plot(const int iter, int population, bool **mesh);
};
