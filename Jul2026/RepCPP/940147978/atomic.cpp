#include <iostream>
#include <omp.h>

int main() {
    int sum = 0;  // Shared variable

    #pragma omp parallel for
    for (int i = 1; i <= 10; i++) {
        #pragma omp atomic
        sum += i;  // Atomic addition prevents race conditions
    }

    std::cout << "Sum = " << sum << std::endl;  // Expected output: 55
    return 0;
}
