#include <chrono>
#include <iostream>
#include <omp.h>
using namespace std;

/*

omp parallel for :
syntax :
    #pragma omp parallel for {
        for(<work>;<work>;<work>){
            <stmts>
        }
    }

things to take care of (Programmer responsiblity):
    1) omp parallel for should not be used with data dependencies or when lhs
and rhs of any computation might alais, it could introduce data hazard. for ex :
c[i] = a[i] + c[i-1], this has loop carried dependence. Hence omp parallel for
can not be used.

    2) Threads should not write to shared data space without using thread sync,
like atomic or critical section.

    3) in parallel for construct no allocation of jobs required manually, omp
does it by default.

    4) Threads will join at the end of for loop. Spwan at the beginning.

Data sharing : Reduction (only (commutative and associative) opreations allowed for reduction where order does not matter)
example : summation of arrays is a common operation. 
Syntax : 
    #pragma omp parallel for reduction (+: <var>)
    for(<work>;<work>;<work>){
        <var> += <data_space>
    }
    
    
    so what is the use of this? is it the same as omp parallel for? Yes and No. 
    This reduction opreation with parallel for loops auto reduction, using the reduction <var> 
    say summing of 1000 elements of an array into reduction variable sum with 10 threads, it does allocation of 100 elements per thread and a thread local variable holds the summation result and instead of collecting the final output in variable sum linearly, it collets in the tree format. And adds the subtree computations using omp parallel constructs, As follows. Which is faster and more efficent for larger data space sizes. 

    s0  s1  s2  s3  s4  s5  s6  s7  s8  s9
      s01     s23     s45     s67     s89 
          s0123           s4567     s89 
                s01234567       s89
                       s0123456789
*/

long numSteps = 1000000;
double deltaX;

void piComputationUsingReduction() {
  double x, pi = 0.0;
  deltaX = 1.0 / (double(numSteps));
#pragma omp parallel for reduction(+ : pi)
  for (int i = 0; i < numSteps; i++) {
    x = (i + 0.5) * (deltaX);
    double fx = (4.0 / (1 + x * x)) * deltaX;
    pi = pi + fx;
  }

  cout << pi;
}

/*
    # for Loop : Scheduling Policy
    which iterations for which threads?

    This can cause load imbalance. If not done properly
    Synatx : 
        schedule(static,<noOfConsecutiveIters>) -> Round robin policy

        schedule(dynamic,<noOfConsecutiveIters>) -> First come first serve basis

        schedule(auto) -> omp decides policy

        schedule(runtime) -> let omp runtime decide.

    
    # Syncronization : Barrier
        To join all threads.
    Duing parallel execution we may need to join the threads this is where Barrier is used. There
    are implicit and explict Barriers.

    explicit barrier syntax : #pragma omp barrier

    There is implict barrier at the end of each work sharing construct, like omp parallel for.
    ex : 
        #pragma omp for
        for(<>;<>;<>){

        } -> end of implicit barrier, here all threads will be joined.
    

    no wait : If we do not want the threads to wait for all other threads to finish at the end of an implicit barrier ( at the end of work sharing construct ) we can specify 'nowait' keyword to allow the threads to proceed without stopping at the barrier.
    ex : 
        #pragma omp for nowait
        for(<>;<>;<>){

        } -> no implicit barrier here as nowait is specified for this for loop.

    and there is always an implicit barrier at the end of a parallel reigon


    # master construct : a block only executed by master thread. others will skip that block.
    syntax :
        #pragma omp master
        {
            doStuff();
        } -> no implicit barrier, since only master should do this computation.

    #single construct : only one thread to execute this part, no not nececerilly master thread any one.
    syntax : 
        #pragma omp single
        {
            doStuff();
        } -> implict barrier exists.

    # sections : openmp sections.
    This is like switch statement so one thread will pick up one section provided no true data
    dependencies exist, and all are independent. Useful where task parallism exits, for example DAG computations, and dependency graph computations. As shown below, X,Y and Z can work
    independently and once all are finished only then A and B can work. iw A and B depend on X,Y and Z.

            X       Y       Z
                \  |   /
                A    B


    syntax :

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            computeX();
            
            #pragma omp section
            computeY();
            
            #pragma omp section
            computeZ();
        } -> implit barrier.

        #pragma omp sections
        {
            #pragma omp section
            computeA();
            
            #pragma omp section
            computeB;
        }
    }
    # Tasks : open mp tasks.


*/

int fib(int n){
    // A master thread enters this call, and on hitting omp task a fork of the thread is created for x and y so two threads independently compute f(n-1) and f(n-2) and on taskwait they are joined to sync.
    int x,y; 
    if(n<2)
        return n;
    #pragma omp task shared(x)
    // this call will be invoked and no implict barrier at the end of omp task, hence some other thread will proceed with fib(n-2) computation, so both these computations can be run in parallel. Omp may delay, the task execution, it depends on omp runtime. Important Note : by Default x and y are private to tasks, explicit shared shoule be mentioned for correctness of the program. Why did we not use omp parallel here? reason each thread will redundantly try to compute both the calls f(n-1) and f(n-2) which is wasted work, what task does really well, is it assigns one thread to a call say f(n-1) and other thread to f(n-2) and both can run independently in parallel, and after both calls are done computing taskwait will provide a barrier for both the values of x and y to be consistant and correct.
    x = fib(n-1);
    #pragma omp task shared(y)
    y = fib(n-2);
    #pragma omp taskwait 
    return x+y;
}


int main() { cout<<fib(10); }