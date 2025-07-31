#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    int n = 210;

    int x = 0;
    #pragma omp parallel for reduction(+: x)
    for (int i = 0; i < n; ++i) {
        x += 2 * i + 1;
    }

    printf("%d ^ 2 = %d\n", n, x);

    return 0;
}