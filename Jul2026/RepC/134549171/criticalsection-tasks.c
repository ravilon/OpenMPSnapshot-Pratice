/*
 * If a lock is held across a task scheduling point, no attempt should be made to acquire the same lock in any code that may be interleaved.  Otherwise, a deadlock is possible.
 *
 * In the example below, suppose the thread executing task 1 defers task 2.  When it 
 * encounters the task scheduling point at task 3, it could suspend task 1 and begin task 2 
 * which will result in a deadlock when it tries to enter critical region 1. 
*/ 


void crit()
{
   #pragma omp task 
   { //Task 1
       #pragma omp task 
       { //Task 2
            #pragma omp critical //Critical region 1 
            {/*do work here */ }       
       }
       #pragma omp critical //Critical Region 2
       {
           //Capture data for the following task
           #pragma omp task
           { /* do work here */ } //Task 3
       }
   }
}

int main()
{
crit();
return 0;
}
       
