#include <iostream>
#include <omp.h>

int main()
{
    // Fork a team of threads
    #pragma omp parallel
    {
        std::cout << "Hello, World!" << std::endl;
    }
}