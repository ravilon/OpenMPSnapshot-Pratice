#include "omp.h"

#include <iostream>

/**
  * @brief File demoing concept of advanced mechanisms to be used instead of criticals
  * Parallel tasks often produce some quantity that needs to be summed or otherwise combined - reduction is the term which denotes this combined result. Whilst one way of doing this would be to simply setup a critical section (see schedueling.cpp) to avoid a data race, this is a pretty poor approach and it's too mucb serialism in a parallel sectiom
    * The reduction clause in OpenMP gives each a thread-private copy of the reduction variable (of the same name). At the end of the parallel region, the values are then combined (using the specified reduction operation) and are saved back into the original variable pass into the reduction clause
    * Syntax: #pragma omp parallel (for) reduction(<reduce-operation> : <reduction_var>)
      * ...where the reduce-operation is either: the predefined: +, *, -, max, min, & (bitwise AND), && (boolean AND), | (bitwise OR), || (boolean OR), ^ (bitwise XOR) OR your own function
        * note: your own reduction function must be declared in a weird openmp syntax: #pragma omp declare reduction (<func_name> : <type> : <operation>)
            * ... where type is the data type of the input variable (set to name omp_in) to be combined with the overall reduction (set to name omp_out) and <operation> is the line of code telling it what to do
  * atomic operations are memory operations turned into atomic ones (things which complete at once as a single non breakable step) - this alleviates the need of setting up critical sections, as the thread will not switch, leaving the intended operation incomplete, before fully executing
    * Syntax: #pragma omp atomic <semantics>
        * ...where the semantics of atomicity are 4: read, write, update, capture
    * Note: these are restricted to certain datatypes
  */

#pragma omp declare reduction (add_func : int : \
    omp_out = omp_in + omp_out \
)

int main()
{
    /* reduction */
    int reduction = 0 ;
    #pragma omp parallel for reduction(add_func : reduction)
    for(std::size_t i = 0 ; i < 10 ; ++i)
    {
        reduction = i ;
    }
    std::cout << "I should have a value of 0+1+2+3+4+5+6+7+8+9 (45)!: " << reduction << std::endl ;

    /* atomics */
    int total = 0 ;
    #pragma omp parallel for
    for(std::size_t i = 0 ; i < 10 ; ++i)
    {
        #pragma omp atomic update
        ++total ;
    }
    std::cout << "I should have the value of 1+1+1+1+1+1+1+1+1+1+1 (10): " << total << std::endl ;

    //
	return 0 ;
}
