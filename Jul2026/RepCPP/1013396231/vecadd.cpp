#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
/*#define N     100000000*/

int main (int argc, const char **argv)
{ /*array from 1K to 1G*/
std::vector<int> arr_sizes = {1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 
1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 
128000000, 256000000, 512000000, 1000000000
};
for (int N : arr_sizes){                          
std::vector<float> a(N), b(N), c(N);
/* Some initialisation */

for (int i=0; i < N; i++)
a[i] = b[i] = i * 1.0;

omp_set_num_threads(4);
auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
for (int i=0; i < N; i++)
c[i] = a[i] + b[i];
auto t2 = std::chrono::high_resolution_clock::now();
std::cout << "Array size: " << N << ","
<< " and Vector addition took "
<< std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
<< " milliseconds \n";
}
return 0;
}

