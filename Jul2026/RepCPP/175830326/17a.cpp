#include <cstdio>
#include <iostream>
#include <omp.h>

int main(int argc, char** argv)
{
	int counter, number;

	#pragma omp parallel private(counter, number)
	{
		counter = omp_get_num_threads();
		number = omp_get_thread_num();
		if (number == 0) {
			std::printf("ThreadCount = %d\n", counter);
		} else {
			std::printf("ThreadNumber = %d\n", number);
		}
	}
	std::cout << "Execution finished" << std::endl;

	return 0;
}
