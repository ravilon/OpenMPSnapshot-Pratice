#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <locale>

void printHelloFromThread(int threadNum) {
    printf("Hello World! От потока %d\n", threadNum);
}

void printHelloFromThreadCout(int threadNum) {
    std::cout << "Hello World! От потока " << threadNum << std::endl;
}

int main(int argc, char* argv[]) {
    std::locale::global(std::locale(""));

    int maxThreads = omp_get_max_threads();
    std::cout << "Максимальное количество потоков: " << maxThreads << std::endl;

    int numThreads;
    std::cout << "Введите количество потоков Х: ";
    std::cin >> numThreads;

    omp_set_num_threads(numThreads);

    printf("\n\nВывод с помощью printf\n");
    #pragma omp parallel
    {
        printHelloFromThread(omp_get_thread_num());
    }

    printf("\n\nВывод с помощью cout\n");
    #pragma omp parallel
    {
        printHelloFromThreadCout(omp_get_thread_num());
    }

    return 0;
}
