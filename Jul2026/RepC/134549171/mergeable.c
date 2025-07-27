/*
 *  The addition of this clause allows the implementation to reuse the data environment of the parent task for the task inside it.
 *  Thus, the result of the execution may differ depending on whether the task is merged or not. 
 *  When a mergeable clause is present on a task construct, then the implementation may	choose to generate a merged task instead. If a merged	task is	generated, then	the behavior is	as though there	was no task directive at all
*/


#include<stdlib.h>
#include<sched.h>
#include<stdio.h>
#include<omp.h>

//In this example, the use of the mergeable clause is safe. As x is a shared variable the outcome does not depend on whether or not the task is merged (that is, thetask will always increment the same variable and will always compute the same value for x).

void func1()
{
 int x = 2;
 #pragma omp task shared(x) mergeable
 {
 x++;
 }
 #pragma omp taskwait
 printf("VALUE : %d\n",x); // prints 3
}


//incorrect use of the mergeable clause.
//In this example, the created task will access different instances of the variable x if the task is not merged, as x is firstprivate, but it will access the same variable x if the task is merged. 
//As a result, the behavior of the program is unspecified and it can print two different values for x depending on the decisions taken by the implementation.

void func2()
{
 int x = 2;
 #pragma omp task mergeable
 {
 x++;
 }
 #pragma omp taskwait
 printf("VALUE : %d\n",x); // prints 2 or 3
}

int main()
{
   func1();
   func2();
}
