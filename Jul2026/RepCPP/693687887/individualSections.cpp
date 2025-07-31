#include <iostream>
#include <omp.h>
#include <cmath>

static constexpr int N = 20000;
static constexpr double tau = 0.0001;
static constexpr double eps = 0.000001;
static constexpr int PROGRAM_LAUNCHES_NUM = 20;

double *createAndReleaseMatrixA() {
    double *matrix = new double[N * N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                matrix[i * N + j] = 2.0;
            } else {
                matrix[i * N + j] = 1.0;
            }

        }
    }

    return matrix;
}

double *createAndReleaseVectorWithNum(double value) {
    double *vector = new double[N];
    for (int i = 0; i < N; ++i) {
        vector[i] = (value);
    }
    return vector;
}

void printVector(double *vector) {
    std::cout << "( ";

    for (int i = 0; i < N; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << ")";
}


int main() {
    for (int k = 1; k < PROGRAM_LAUNCHES_NUM; ++k) {
        double *matrixA = createAndReleaseMatrixA();
        double *vectorB = createAndReleaseVectorWithNum(N + 1.0);
        double *vectorX = createAndReleaseVectorWithNum(0);
        double *previousIterationVectorX = new double[N]();

        double normOfB = 0;
        double normOfV = 0;
        double newNormCriteria = 0;
        bool criteria = false;


        omp_set_dynamic(0);
        omp_set_num_threads(k);
        double startTime = omp_get_wtime();

#pragma omp parallel for schedule(static) reduction(+:normOfB)
        for (int i = 0; i < N; ++i) {
            normOfB += vectorB[i] * vectorB[i];
        }

        normOfB = sqrt(normOfB);


        while (!criteria) {
#pragma omp parallel for schedule(static) reduction(+:normOfV)
            for (int i = 0; i < N; ++i) {
                double oneCeilValueOfX = 0;

                const double *oneRowOfMatrix = matrixA + i * N;
                for (int j = 0; j < N; ++j) {
                    oneCeilValueOfX += oneRowOfMatrix[j] * vectorX[j];
                }

                oneCeilValueOfX -= vectorB[i];

                previousIterationVectorX[i] = vectorX[i] - oneCeilValueOfX * tau;
                normOfV += oneCeilValueOfX * oneCeilValueOfX;
            }

            std::swap(vectorX, previousIterationVectorX);

            normOfV = sqrt(normOfV);
            newNormCriteria = normOfV / normOfB;
            criteria = newNormCriteria < eps;
            normOfV = 0;
        }


        double end_time = omp_get_wtime();

        std::cout << "Time passed: " << end_time - startTime << " seconds. Threads used: " << omp_get_max_threads()
                  << std::endl;

        delete[] matrixA;
        delete[] vectorX;
        delete[] vectorB;
        delete[] previousIterationVectorX;
    }
    return 0;
}