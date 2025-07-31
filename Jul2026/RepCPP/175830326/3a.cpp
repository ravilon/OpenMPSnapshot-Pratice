#include <iostream>

int main()
{
	std::cout << "Area sequential 1" << std::endl;
	int counter = 0;

	#pragma omp parallel reduction (+: counter)
	{
		std::cout << "Area parallel\n";
		++counter;
	}
	std::cout << "Area sequential 2, counter = " << counter << std::endl;

	return 0;
}
