#include "thread.h"
#include "field_manager.h"
#include <cstdlib>
#include <vector>
#include <iostream>
#include <omp.h>

Thread::Thread(int threadNumber, fieldType myInitialPart, fieldType myInitialBorders,
   int numberOfIterations, //pthread_cond_t* stopped, pthread_mutex_t* stopMutex,
    FieldManager& manager, int numberOfThreads):
manager(manager),
threadNumber(threadNumber),
numberOfIterations(numberOfIterations),
numberOfThreads(numberOfThreads) {
  waiting = true;
  //this->stopped = stopped;
  //this->stopMutex = stopMutex;
  chunkHeight = myInitialPart.size();
  chunkWigth = myInitialPart[0].size();
  myPartWithBorders.push_back(myInitialBorders[0]);
  for (int i = 0; i < chunkHeight; i++) {
    myPartWithBorders.push_back(myInitialPart[i]);
  }
  myPartWithBorders.push_back(myInitialBorders[1]);
  //leftSemaphore = (sem_t*) malloc(sizeof(sem_t));
  //rightSemaphore = (sem_t*) malloc(sizeof(sem_t));
  //leftControlSemaphore = (sem_t*) malloc(sizeof(sem_t));
  //rightControlSemaphore = (sem_t*) malloc(sizeof(sem_t));
  //sem_init(leftSemaphore, 0, 0);
  //sem_init(rightSemaphore, 0, 0);
  //sem_init(leftControlSemaphore, 0, 0);
  //sem_init(rightControlSemaphore, 0, 0);
  currentIteration = 0;
  leftThread = NULL;
  rightThread = NULL;
  threadDescriptor = 1;
  cancelled = false;
}

void Thread::getAdjacentThreads(Thread& leftThread, Thread& rightThread) {
  this->leftThread = &leftThread;
  this->rightThread = &rightThread;
}

void Thread::create() {
  //pthread_create(&threadDescriptor,NULL,&Thread::runInThread,this);
}

fieldType Thread::getComputedPart() {
  fieldType computedPart;
  for (int i = 0; i < chunkHeight; i++) {
    computedPart.push_back(myPartWithBorders[i+1]);
  }
  return computedPart;
}
ll Thread::getCurrentIteration() {
  return currentIteration;
}

void Thread::cancel(bool wait) {
  waiting = wait;
  cancelled = true;
  //pthread_join(threadDescriptor, NULL);
  //pthread_detach(threadDescriptor);
  //destroySemaphores();
}
void Thread::destroySemaphores() {
  /*sem_destroy(leftControlSemaphore);
  free(leftControlSemaphore);
  sem_destroy(rightControlSemaphore);
  free(rightControlSemaphore);
  sem_destroy(leftSemaphore);
  free(leftSemaphore);
  sem_destroy(rightSemaphore);
  free(rightSemaphore);*/
}


void* Thread::runInThread(void *thisThread) {
  Thread* t = (Thread*) thisThread;
  t->run();
  return NULL;
}
void Thread::updateIterations(ll numberOfIterations) {
  this->numberOfIterations = numberOfIterations + currentIteration;
}
void Thread::run() {
  omp_set_num_threads(numberOfThreads);

  //while (!cancelled || waiting) {
    //pthread_mutex_lock(stopMutex);
    //bool out = false;
    //while (manager.wasStopped()) {
      /*if (!out) {
        std::cout << threadNumber << ": stopped" << std::endl;
        out = true;
      }*/
      //pthread_cond_wait(stopped, stopMutex);
    //}

    //pthread_mutex_unlock(stopMutex);

    while(currentIteration < numberOfIterations && !cancelled && !manager.wasStopped()) {
        //std::cout /*<< threadNumber */<< ": current iteration is " << currentIteration << std::endl;
        oneIteration();
        currentIteration++;
    }
    /*if (currentIteration == numberOfIterations && cancelled) {
      waiting = false;
    }
  }*/
  //std::cout << threadNumber << " :cancelled" << std::endl;
}
void Thread::oneIteration() {
  int sum;
  fieldType myNewPart(myPartWithBorders);
  #pragma omp parallel for
  for (ll i = 1; i < chunkHeight + 1; i++) {
    for (ll j = 0; j < chunkWigth; j++) {
      sum = numberOfNeighbours(i, j);
      if (myPartWithBorders[i][j]) {
        myNewPart[i][j] = (sum == 2) || (sum == 3);
      } else {
        myNewPart[i][j] = (sum == 3);
      }
    }
  }
  myPartWithBorders = myNewPart;
  ll size = myPartWithBorders.size();
  std::vector<bool> tmp = myPartWithBorders[1];
  myPartWithBorders[0] = myPartWithBorders[size - 2];
  myPartWithBorders[size - 1] = tmp;
  //std::cout << threadNumber << ": computed my part" << std::endl;;
  //exchangeBorders();
}

void Thread::exchangeBorders() {
  /*
  //let read my borders.
  std::cout << threadNumber << ": raising semaphores..." <<std::endl;
  sem_post(leftSemaphore);
  sem_post(rightSemaphore);

  std::cout << threadNumber << ": waiting for left thread" << std::endl;
  //read adjacent borders.
  sem_wait(leftThread->rightSemaphore);
  myPartWithBorders[0] = leftThread->myPartWithBorders[leftThread->chunkHeight];
  std::cout << threadNumber << ": raising control semaphore of left thread" << std::endl;
  sem_post(leftThread->rightControlSemaphore);

  std::cout << threadNumber << ": waiting for right thread" << std::endl;
  sem_wait(rightThread->leftSemaphore);
  myPartWithBorders[chunkHeight + 1] = rightThread->myPartWithBorders[1];
  std::cout << threadNumber << ": raising control semaphore of right thread" << std::endl;
  sem_post(rightThread->leftControlSemaphore);

  std::cout << threadNumber << ": waiting for my control semaphores" << std::endl;
  //check that my borders were read.
  sem_wait(rightControlSemaphore);
  sem_wait(leftControlSemaphore);
  std:: cout << threadNumber << ": exchanged" << std::endl;
  */
}

int Thread::numberOfNeighbours(ll i, ll j) {
  int sum = 0;
  ll p,q;
  for (int deltaI = -1; deltaI < 2; deltaI++) {
    for (int deltaJ = -1; deltaJ < 2; deltaJ++) {
      p = i+deltaI;
      q = j+deltaJ;
      if (p >= chunkHeight + 2) {
        p = 0;
      } else if (p < 0) {
        p = chunkHeight + 1;
      }
      if (q >= chunkWigth) {
        q = 0;
      } else if (q < 0) {
        q = chunkWigth - 1;
      }
      sum += myPartWithBorders[p][q];
    }
  }
  //because it was added to sum in for.
  sum -= myPartWithBorders[i][j];
  return sum;
}
