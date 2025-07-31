#include <iostream>
#include <omp.h>
#include <stdio.h>

using namespace std;

int main(int argc, char const *argv[]) {

    // printing the number of disponible threads
    printf("Maximum number of threads: %d\n", omp_get_max_threads());

    // doing a test if this program's is parallel
    // the expected answer is 'no'
    int parallel = omp_in_parallel() ? 1 : 0;
    printf("Am I in a parallel construct?\n");
    if (parallel) cout << "Yes\n"; else cout << "No\n";
    if (parallel) cout << "Thread: " << omp_get_thread_num() << "\n";

    // now we have the number of disponible threads as the number of iterations 
    // to be effectuated. A parallel region begins with the #pragma below. So, if 
    // we ask again if this program's part is in parallel, we should we receive 'yes'
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++) {
        parallel = omp_in_parallel() ? 1 : 0;
        printf("Am I in a parallel construct?\n");
        if (parallel) cout << "Yes\n"; else cout << "No\n";
        if (parallel) cout << "Thread: " << omp_get_thread_num() << "\n";
        cout << "Testing openMP!" << '\n';
    }

    return 0;
}
