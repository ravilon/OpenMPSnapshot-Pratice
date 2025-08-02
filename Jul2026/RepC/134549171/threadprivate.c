/*
 *  A threadprivate variable can be modified by another task that is executed by the same thread. 
 *  The parts of these task regions in which tp is modified may be executed in any order so the resulting value of var can be either 1 or 2.
*/

#include<stdlib.h>
#include<stdio.h>
#include<omp.h>

int tp;
#pragma omp threadprivate(tp)
int var;
void work()
{
#pragma omp task
 {
 /* do work here */
#pragma omp task
 {
 tp = 1;
 /* do work here */
#pragma omp task
 {
 /* no modification of tp */
 }
 var = tp; //value of tp can be 1 or 2
 }
 tp = 2;
 }
}

int main()
{
work();
printf("VALUE OF VAR IS : %d\n",var);
return 0;
}
