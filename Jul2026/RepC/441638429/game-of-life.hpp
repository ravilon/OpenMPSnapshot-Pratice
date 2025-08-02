#pragma once

#include "general_utils.hpp"
#include "plotter.hpp"
#include "real_rand.hpp"

typedef enum GameType { DEFAULT, BLOCK } game_e;

/**
 * Game of Life.
 *
 * The constructor sets up the game parameters, and runs the program.
 */
class GameOfLife : public Project {
 private:
  // parameters
  int nx;              // num. squares in X axis
  int ny;              // num. squares in Y axis
  int numthreads;      // number of threads
  int maxiter;         // max. no of iterations
  float prob;          // probability of placing a cell in world generation
  bool isPlotEnabled;  // enable GNU plotting
  game_e game;         // game type, such as random / glider etc.

  // game data
  GameOfLifePlotter *plotter = NULL;  // GNU plotting API
  int population = 0;                 // population of the current world
  bool **currWorld = NULL;            // current world
  bool **nextWorld = NULL;            // next world
  bool **tmpWorld = NULL;             // temporary pointer for swapping
  float runtimeMS = 0;

  // auxillary function for world population
  int populateCurrentWorld();

 public:
  GameOfLife(int nx, int ny, int numthreads, int maxiter, float prob, bool isPlotEnabled, game_e game);
  ~GameOfLife();

  /**
   * Serial implementation of the Game of Life.
   */
  void serial();

  /**
   * Parallel implementation of the Game of Life using OpenMP. The plotting and calculations are done in parallel as
   * OpenMP tasks. Within the calculation task, multiple threads process the game in parallel.
   */
  void parallel();

  /**
   * Print information about the game.
   */
  void printParameters(runtype_e runType);
};