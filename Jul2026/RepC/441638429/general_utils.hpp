#pragma once

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

// @note assumes existence of argv, ac and s variables
#define MATCH_ARG(s) (!strcmp(argv[ac], (s)))

// Enum for Running type of the project
typedef enum RunningType { SERIAL, PARALLEL } runtype_e;

/**
 * A simple stopwatch.
 */
class Stopwatch {
 private:
  double startTime = 0;
  double stopTime = 0;
  bool isRunning = false;
  double getTime() {
    struct timeval TV;
    struct timezone TZ;
    const int RC = gettimeofday(&TV, &TZ);
    if (RC == -1) {
      printf("ERROR: Bad call to gettimeofday\n");
      return (-1);
    }
    return (((double)TV.tv_sec) + 1.0e-6 * ((double)TV.tv_usec));
  }

 public:
  void start() {
    if (!isRunning) {
      startTime = getTime();
      isRunning = true;
    }
  }
  double stop() {
    if (isRunning) {
      stopTime = getTime();
      isRunning = false;
      return stopTime - startTime;
    } else {
      return 0;
    }
  }
};

/**
 * Base class for all projects.
 */
class Project {
 private:
  Stopwatch* sw = NULL;

 protected:
  virtual void serial() = 0;
  virtual void parallel() = 0;
  virtual void printParameters(runtype_e runType) = 0;

 public:
  double run(runtype_e runType) {
    // print runtime info
    this->printParameters(runType);

    // start stopwatch
    this->sw->start();

    // run the program
    if (runType == SERIAL) {
      this->serial();
    } else if (runType == PARALLEL) {
      this->parallel();
    }

    // stop stopwatch and report elapsed time
    return this->sw->stop();
  }

  Project() { this->sw = new Stopwatch(); }

  ~Project() { delete this->sw; }
};

/**
 * A tiny utility function to wait for arbitrary user input.
 * @param msg custom message prompt
 */
inline void press_any_key(const char* msg = "Press any key to continue.") {
  printf("%s\n", msg);
  getchar();
}