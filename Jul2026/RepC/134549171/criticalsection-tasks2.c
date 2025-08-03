/*
* In the following example, lock is held across a task scheduling point.
*
* However, according to the scheduling restrictions, the executing thread can't begin executing one of the non-descendant tasks that also acquires 
* lock before the task region is complete.  
*
* Therefore no deadlock is possible.
*/


#include <omp.h>
#include<stdlib.h>
#include<stdio.h>

void work() {
omp_lock_t lock;
omp_init_lock(&lock);
#pragma omp parallel
{
int i;
#pragma omp for
for (i = 0; i < 100; i++) {
#pragma omp task 
{ 
//lock is shared by default in the task
omp_set_lock(&lock);
// Capture data for the following task
#pragma omp task
// Task Scheduling Point 1
{ /* do work here */ }
omp_unset_lock(&lock);
}
}
}
omp_destroy_lock(&lock);
}

int main()
{
work();
return 0;
}





















