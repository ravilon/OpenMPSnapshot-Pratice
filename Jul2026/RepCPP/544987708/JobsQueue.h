//
// Created by davide on 04/10/22.
//

#ifndef N_GRAM_JOBSQUEUE_H
#define N_GRAM_JOBSQUEUE_H

#include <omp.h>
#include <iostream>
#include <vector>


class Job {
public:
    std::vector<std::string> dataChunk;
    Job* next;
};

class JobsQueue {
    /* Queue of jobs for the Producer-Consumer pattern */
public:

    JobsQueue(int numProd=1) {
        Job* tmp = new Job(); // placeholder for the empty queue
        tmp->next = nullptr;
        head = tail = tmp;
        enqueued = dequeued = 0;
        numProducers = numProd;
        omp_init_lock(&headLock);
        omp_init_lock(&tailLock);
    }

    ~JobsQueue() {
        Job* curr = head;
        Job* tmp;

        while (curr != nullptr) {
            tmp = curr;
            curr = curr->next;
            delete(tmp);
        }

        enqueued = dequeued = 0;
        head = tail = nullptr;
        omp_destroy_lock(&headLock);
        omp_destroy_lock(&tailLock);
    }
    void enqueue(const std::vector<std::string> dataChunk) {
        /* Enqueue on the tail */
        Job* tmp = new Job();
        tmp->dataChunk = dataChunk;
        tmp->next  = nullptr;
        omp_set_lock(&tailLock); // we need to update the tail pointer, we don't want any other thread to do so at the same time
        tail->next = tmp;
        tail = tmp;
        enqueued++;
        omp_unset_lock(&tailLock);
    }
    bool dequeue(std::vector<std::string>& dataChunk) {
        /* Dequeue from the head */
        omp_set_lock(&headLock);
        Job* tmp = head;
        Job* newHead = tmp->next;
        if (newHead == nullptr) {
            // Case of empty queue
            omp_unset_lock(&headLock);
            return false;
        } else {
            // Case of dequeue successful
            dataChunk = newHead->dataChunk;
            head = newHead;
            dequeued++;
            omp_unset_lock(&headLock);
            delete tmp;
            return true;
        }
    }
    bool done() {
    #pragma omp flush(numProducers, enqueued) // as these are shared we make sure we read the updated value
        if ((enqueued-dequeued) == 0 && numProducers== 0)
            return true;
        else
            return false;
    }
    void producerEnd() {
    #pragma omp atomic
        numProducers--;
    }

private:
    Job* head;
    Job* tail;
    omp_lock_t headLock;
    omp_lock_t tailLock;
    int enqueued;
    int dequeued;
    int numProducers;
};

#endif //N_GRAM_JOBSQUEUE_H
