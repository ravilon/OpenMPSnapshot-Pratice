/*
* The following example illustrates the difference between the if and the final clauses.
*
* The if clause has a local effect. In the first nest of tasks, the one that has the if clause will be undeferred 
* but the task nested inside that task will not be affected by the if clause and will be created as usual. 
*
* Alternatively, the final clause affects all task constructs in the final
* task region but not the final task itself. In the second nest of tasks, the nested tasks will be created as 
* included tasks. Note also that the conditions for the if and final clauses are usually the opposite.
*
*/


#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
void bar()
{
}
void foo ( )
{
int i;
#pragma omp task if(0)  // This task is undeferred
{
#pragma omp task     // This task is a regular task
for (i = 0; i < 3; i++) {
#pragma omp task     // This task is a regular task
bar();
}
}
#pragma omp task final(1) // This task is a regular task
{
#pragma omp task  // This task is included
for (i = 0; i < 3; i++) {
#pragma omp task     // This task is also included
bar();
}
}
}

int main()
{
foo();
return 0;
}



