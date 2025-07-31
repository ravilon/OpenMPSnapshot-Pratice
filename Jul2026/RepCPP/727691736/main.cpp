
// OpenMP header
#include <omp.h>

#include <array>
#include <iostream>

int main(int argc, char* argv[])
{
    const int n = 10000;
    std::array<int, n> arr;

    // If there is no number of threads defined OpenMP will create max number of possible threads.
    // So optionally one can define number of threads with #pragma omp parallel for num_threads(4)
#pragma omp parallel for
    for ( int i = 0; i<n; ++i){
        arr[i] = i + 1;
    }


    // Proof of concept: Stream is printed unorder and mixed since multi threading
#pragma omp parallel for
    for(auto elem: arr){
        std::cout << "Array Entry: " << elem << std::endl;
    }


    // TODO: Explore different features for OpenMP to check what is the most suitable feature for our Project

    return 0;
}
