#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

int main(int argc, char *argv[]) {
    int width, height, maxIterations, numThreads;
    double xMin, xMax, yMin, yMax;

    scanf("%d %d %lf %lf %lf %lf %d %d", &width, &height, &xMin, &xMax, &yMin,
          &yMax, &maxIterations, &numThreads);

    printf("-> Starting mandelbrot generation in \"Parallel\"\n");
    printf("-> Resolution: %dx%d\n", width, height);
    printf("-> Complex plane limit (x): [%lf, %lf]\n", xMin, xMax);
    printf("-> Complex plane limit (y): [%lf, %lf]\n", yMin, yMax);
    printf("-> Max number of iterations: %d\n", maxIterations);
    printf("-> Number of Threads: %d\n", numThreads);

    int *iterationsSpace = create_iterations_space(width, height);

    double *xLinearSpace = create_linear_space(xMin, xMax, width);
    double *yLinearSpace = create_linear_space(yMin, yMax, height);

    int grainSize = 100;
    int currentPos = 0;

    double programStart = omp_get_wtime();

#pragma omp parallel num_threads(numThreads) shared(currentPos)

    {
        double start = omp_get_wtime();
        int num = omp_get_thread_num();

        int myPos = 0;
        while (1) {
#pragma omp critical
            {
                myPos = currentPos++;
            }

            if (myPos * grainSize >= width) {
                break;
            }

            int upperBound = (myPos + 1) * grainSize;
            upperBound = upperBound > width ? width : upperBound;

            complex z, c;
            int iterations = 0;
            short int escaped = 0;
            for (int i = grainSize * myPos; i < upperBound; i++) {
                for (int j = 0; j < height; j++) {
                    escaped = 0;
                    iterations = 0;

                    z.real = 0;
                    z.imag = 0;

                    c.real = xLinearSpace[i];
                    c.imag = yLinearSpace[j];

                    while (iterations < maxIterations) {
                        if (abs_complex(z) > 2) {
                            escaped = 1;
                            break;
                        }
                        z = complex_squared(z);

                        z.real = z.real + c.real;
                        z.imag = z.imag + c.imag;

                        iterations++;
                    }

                    iterationsSpace[j * width + i] = escaped ? iterations : 0;
                }
            }
        }

        double end = omp_get_wtime();
        printf("\t-> Thread #%d done in %.6f seconds!\n", num, end - start);
    }

    double programEnd = omp_get_wtime();

    // logExperiment(numThreads, maxIterations, programEnd - programStart);

    const char *outputName = "mandelbrot.png";

    save_result_to_png(outputName, iterationsSpace, width, height,
                       maxIterations);

    free(iterationsSpace);
    free(xLinearSpace);
    free(yLinearSpace);

    return 0;
}