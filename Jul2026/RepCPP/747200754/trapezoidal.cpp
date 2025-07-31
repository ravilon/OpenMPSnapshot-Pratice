#include <omp.h>

#include <cmath>
#include <iomanip>
#include <iostream>

inline constexpr float f(const float& x) noexcept {
    return 2.0f / (1.0f + std::pow(x, 4));
}

int main(int argc, char const* argv[]) {
    const int a = 0;
    const int b = 1;
    const long n = 1024000000;
    const double h = (b - a) / (static_cast<const double>(n));

    double integral = (f(a) + f(b)) / 2.0f;
    double x = static_cast<const double>(a);

    const double start_time = omp_get_wtime();

#pragma omp parallel for reduction(+ : integral) reduction(+ : x) schedule(static)
    for (long i = 1; i <= n; ++i) {
        x += h;
        integral += f(x);
    }

#pragma omp barrier
    const double end_time = omp_get_wtime();

    integral *= h;

    std::cout
        << "With n = " << n
        << std::setprecision(16)
        << " trapezoids, estimate: " << integral
        << '\n';

    std::cout << "Time: " << end_time - start_time << '\n';

    return 0;
}
