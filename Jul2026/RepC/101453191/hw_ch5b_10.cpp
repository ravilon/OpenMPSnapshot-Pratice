#include <math.h> 
#include <omp.h> 
#include <iostream> 
#include <string> 
int main(int argc, char* argv[]) { double sTime, eTime; int thread_count = 12, n = 10000000; 
if(argc == 2) thread_count = std::stoi(argv[1]); sTime = omp_get_wtime(); double my_sum = 0.0; 
for (int i = 0; i < n; i++) 
	#pragma omp atomic 
my_sum += sin(i); eTime = omp_get_wtime(); std::cout << "Time of single critical " << eTime - sTime << "\n";
sTime = omp_get_wtime(); 
#pragma omp parallel num_threads(thread_count) 
{ double my_sum = 0.0; for (int i = 0; i < n; i++) 
	#pragma omp atomic 
my_sum += sin(i); 
} 
eTime = omp_get_wtime(); std::cout << "Time of double critical " << eTime - sTime << "\n"; return 0;}