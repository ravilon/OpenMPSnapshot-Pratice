#include <iostream>
#include <random>
#include <ctime>
#include <omp.h>

const int TOTAL_POINTS = 100000;

// Serial Monte Carlo
double monte_carlo_serial()
{
    int points_inside_circle = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < TOTAL_POINTS; i++)
    {
        double x = dis(gen);
        double y = dis(gen);

        if (x * x + y * y <= 1.0)
        {
            points_inside_circle++;
        }
    }

    return 4.0 * points_inside_circle / TOTAL_POINTS;
}

// Parallel Monte Carlo
double monte_carlo_parallel()
{
    int points_inside_circle = 0;

#pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        int local_points_inside_circle = 0;

#pragma omp for
        for (int i = 0; i < TOTAL_POINTS; i++)
        {
            double x = dis(gen);
            double y = dis(gen);

            if (x * x + y * y <= 1.0)
            {
                local_points_inside_circle++;
            }
        }

#pragma omp atomic
        points_inside_circle += local_points_inside_circle;
    }

    return 4.0 * points_inside_circle / TOTAL_POINTS;
}

int main()
{
    double start_time, end_time, time_serial, time_parallel;

    // Serial computation
    start_time = omp_get_wtime();
    double pi_serial = monte_carlo_serial();
    end_time = omp_get_wtime();
    time_serial = end_time - start_time;
    std::cout << "Serial Pi estimation: " << pi_serial << "\n";
    std::cout << "Serial execution time: " << time_serial << " seconds\n\n";

    // Parallel computation
    start_time = omp_get_wtime();
    double pi_parallel = monte_carlo_parallel();
    end_time = omp_get_wtime();
    time_parallel = end_time - start_time;
    std::cout << "Parallel Pi estimation: " << pi_parallel << "\n";
    std::cout << "Parallel execution time: " << time_parallel << " seconds\n\n";

    // Speedup
    double speedup = time_serial / time_parallel;
    std::cout << "Speedup: " << speedup << "\n";

    return 0;
}

