#include "omp.h"

#include <iostream>
#include <string>
#include <cstddef>

/**
  * @brief File demoing how to manipulate how iterations of for loops are divided among threads
  * We can set how OpenMP threads run code they need to process by specifying the `schedule` parameter
    * ie. #pragma omp parallel for schedule(method (, chunk_size))
    * Note: a chunk is known as the number of consequetive processes to run (ie. 25 iteration could be split into chunks of 5 iterations). chunk_size is optional and defaults to a certain value depending on the method used to schedule the code executions.
    * Methods:
        * static: gives each thread roughly the same amount of chunks to process
            * when no chunk_size is specified, the iterations are divided into chunks that are approximately equal in size, with one chunk assigned to each thread
        * dynamic: gives each thread chunks depending on when each are done processing a previous chunk
            * e.g. threadA finishes chunk1, gets chunk2 immediately, finishes chunk2 and gets chunk3 whilst may threadB may only end up processing one
            * When no chunk_size is specified, it defaults to 1 (ie each iteration is a chunk)
        * guided: each thread gets a chunk size which starts as large and gradually decreases
            * this is done in order to progressively improve load balancing (as if a thread suddenly takes all the work, it'll end up being worked on less to give the other threads a chance to take more work)
            * When no chunk_size is specified, it defaults to 1
        * runtime: decision regarding scheduling is deferred until runtime by looking at the OMP_SCHEDULE env variables
            * The schedule kind and chunk_size is set within this variable
            * If this environment variable is not set, the resulting schedule is implementation-defined
            * NOTE: do not specify chunk_size as part of pragma, won't work
        * auto: when specified, the decision regarding scheduling is delegated to the compiler. default action (though writing it makes code clearer)
  * Note: OpenMP is the only modern Parallelism library which still has this schedueling selection - ability to set scheduling is of lesser importance as computer hardware becomes more advanced
  */

int main()
{
    #pragma omp parallel for schedule(static, 2)
    for(std::size_t i = 0 ; i < 32 ; ++i)
    {
        std::cout << "Hi, I should be ran 16 times on: " << omp_get_thread_num() << std::endl ;
    }
    //
	return 0 ;
}
