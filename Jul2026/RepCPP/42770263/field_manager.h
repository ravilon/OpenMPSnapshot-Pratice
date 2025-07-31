#pragma once
#include <vector>
#include <string>
#include <iostream>

class FieldManager {
  std::vector<std::vector<bool> > field;
  long numberOfiterations;
  long currentIteration;
  int numberOfThreads;
  std::ostream& out;
  bool stopped;
public:
  //constructs file manager.
  FieldManager(std::ostream& out);

  //starts Game of Life  with provided distribution.
  void start(const std::string fileName, int numberOfThreads);

  //starts Game of Life with random distribution.
  void start(long width, long height, int numberOfThreads);

  //shows current iteration.
  void status();

  //starts Game of Life and runs it for provided numberOfIterations.
  void run(long numberOfIterations);

  //stops Game of Life.
  void stop();

  //terminates Game of Life.
  void quit(bool useless);
private:
  //parses CSV file.
  void parseCSV(std::string fileName);

  //generates field of provided size.
  void generateField(long wigth, long height);

  //runs one iteration of the game.
  void runIteration();

  //counts live neighbours of a cell.
  int sumOfNeighbours(long i, long j);
};
