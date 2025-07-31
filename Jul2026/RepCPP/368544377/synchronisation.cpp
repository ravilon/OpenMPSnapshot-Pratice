#include <iostream>

namespace omp {
    #include "omp.h"
}

/**
  * @brief File demoing concept of synchoronisation mechanisms
  * Synchronisation constructs allow us to tell threads to bring a certain order to the sequence in which they do things
  * Barriers are the most widely used in OpenMP and it is concept which essentially any thread must await at a determined point till the computation of all threads have reached said point
    * In OpenMP, many of these points are implicit, such as at the end of the parallel region (ie the lexical scope)
        * This includes even if its nested etc.
    * However, you can also both add and remove barriers explicitly
        * Adding syntax: #pragma omp barrier
            * this is place where you want there to be a barrier
        * Removing syntax: #pragma omp parallel ... nowait
            * this 'nowait' clause goes in the setup line of the parallel region
            * This is not recommended however in most instances
    * Critical sections (locked regions) can be setup such that only one thread may enter the processing region at a given time
        * Syntax: #pragma omp critical
            * Note: this lasts till the end of the parallel region the call was made from (for some odd reason, FORTRAN doesnt suffer from this)
        * Note: see advanced_synchronisation.cpp for different ways to achieve what critical sections would be useful for in more time efficient manners
  */

int main()
{
    /* barriers */
    #pragma omp parallel
    {
        int mytid = omp::omp_get_thread_num() ;
        #pragma omp barrier // barrier declared here - no parallel process will able to perform the last line of the region till all have executed the above
        std::cout << "im thread#" << mytid << std::endl ;
    }

    /* critical sections */
    #pragma omp parallel
    {
        #pragma omp critical // forces only one thread to run from here till the end of the critical section
        std::cout << "Hi" << std::endl ; // this should print normally, since both the printing and the flush should both happen before another thread has the oppotunity to call either
    }
    //
    return 0 ;
}
