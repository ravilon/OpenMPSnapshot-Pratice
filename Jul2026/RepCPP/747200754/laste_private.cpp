#include <omp.h>

#include <iostream>

int main(const int argc, char const *argv[]) {
    int a = 0;
    int b = 0;

    omp_set_dynamic(false);
    omp_set_num_threads(4);

#pragma omp parallel
    {
#pragma omp single
        std::cout << "number of thread: " << omp_get_num_threads() << '\n';

#pragma omp for lastprivate(a) schedule(static)
        for (int i = 0; i < 10; ++i) {
#pragma omp critical
            {
                a = b++;
                std::cout
                    << "previous a = " << a++ << '\n'
                    << "current a = " << a << '\n'
                    << "tid = " << omp_get_thread_num() << "\n\n";
            }
        }
    }

    std::cout << "final a = " << a << '\n';

    return 0;
}
