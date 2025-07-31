#pragma once
#include "../common_lib/headers/typedefs.h"
#include <pthread.h>
#include <semaphore.h>
#include <vector>

class FieldManager;

class Thread {
  pthread_t threadDescriptor;
  int threadNumber;
  ll chunkWigth;
  ll chunkHeight;//it's height only of thread's part, without borders;
  Thread* leftThread; // adjacent
  Thread* rightThread; // threads;
  sem_t* leftSemaphore; // tells adjacent threads,
  sem_t* rightSemaphore; //that my borders are ready(or not);
  sem_t* leftControlSemaphore; // checks, that previous reads of borders
  sem_t* rightControlSemaphore; // finished successfully;
  pthread_cond_t* stopped; //condition variable of master, tells thread when to work;
  pthread_mutex_t* stopMutex; // mutex for condition variable;
  ll numberOfIterations;
  ll currentIteration;
  fieldType myPartWithBorders; // has size = chunkHeight + 2;
  FieldManager& manager;
  bool cancelled;
  bool waiting;
public:
  //constructs a new thread wrapper object.
  Thread(int threadNumber, fieldType myInitialPart, fieldType initialBorders,
    int numberOfIterations, pthread_cond_t* stopped, pthread_mutex_t* stopMutex,
    FieldManager& manager);

  //copies references to adjacent threads,important for border exchange.
  void getAdjacentThreads(Thread& leftThread, Thread& rightThread);

  //invokes pthread_create on internal thread.
  void create();

  //invokes cancel on internal thread and performs cleanup.
  void cancel(bool wait);

  //retrieve computed part from thread.
  fieldType getComputedPart();

  //retrieve current iteration number.
  ll getCurrentIteration();

  void updateIterations(ll numberOfIterations);

private:
  static void* runInThread(void* thisThread);
  //runs Game Of Life iterations.
  void run();
  //performs one iteration of Game of Life on provided part of field.
  void oneIteration();

  //sends computed borders to adjacent threads and receives needed borders from them.
  void exchangeBorders();

  //counts live cells around our cell.
  int numberOfNeighbours(ll i, ll j);

  void destroySemaphores();

};
