#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <climits>
#include <bitset>
#include <cmath>

using namespace std;
using namespace chrono;

const int INF = INT_MAX;
constexpr int MAXN = 1000;

int n;
vector<vector<int>> cost;
volatile int bestCost = INF;
int parallelLevel;

int refinedLowerBound(const bitset<MAXN>& mask, int costSoFar, int current) {
    int estimate = costSoFar;
    if ((int)mask.count() == n)
        return estimate + cost[current][0];

    int sum = 0;
    for (int i = 0; i < n; i++) {
        if (!mask.test(i)) {
            int firstMin = INF, secondMin = INF;
            for (int j = 0; j < n; j++) {
                if (i != j && cost[i][j] != 0) {
                    int w = cost[i][j];
                    if (w < firstMin) {
                        secondMin = firstMin;
                        firstMin = w;
                    }
                    else if (w < secondMin) {
                        secondMin = w;
                    }
                }
            }
            sum += firstMin + secondMin;
        }
    }
    return estimate + sum / 2;
}

void tsp(int current, bitset<MAXN> mask, int costSoFar, int depth = 0) {
#pragma omp flush(bestCost)
    int globalBest;
#pragma omp atomic read
    globalBest = bestCost;

    int bound = refinedLowerBound(mask, costSoFar, current);
    if (bound >= globalBest) return;

    if ((int)mask.count() == n) {
        if (cost[current][0] != 0) {
            int tourCost = costSoFar + cost[current][0];
#pragma omp critical
            {
                if (tourCost < bestCost) {
                    bestCost = tourCost;
                }
            }
        }
        return;
    }

    for (int next = 0; next < n; next++) {
        if (!mask.test(next) && cost[current][next] != 0) {
            bitset<MAXN> newMask = mask;
            newMask.set(next);
            int newCost = costSoFar + cost[current][next];

            if (depth < parallelLevel) {
#pragma omp task firstprivate(next, newMask, newCost, depth)
                tsp(next, newMask, newCost, depth + 1);
            }
            else {
                tsp(next, newMask, newCost, depth + 1);
            }
        }
    }

    if (depth < parallelLevel) {
#pragma omp taskwait
    }
}

int main() {

    n = 15;
    cost = {
        {0, 34, 47, 85, 61, 22, 63, 38, 93, 76, 15, 48, 60, 67, 44},
        {34, 0, 59, 64, 39, 46, 80, 77, 50, 42, 26, 73, 85, 58, 30},
        {47, 59, 0, 70, 90, 65, 35, 43, 61, 72, 53, 88, 79, 60, 69},
        {85, 64, 70, 0, 95, 37, 76, 25, 84, 56, 33, 66, 72, 94, 41},
        {61, 39, 90, 95, 0, 50, 74, 69, 57, 33, 22, 80, 47, 59, 73},
        {22, 46, 65, 37, 50, 0, 55, 40, 79, 63, 34, 58, 41, 67, 29},
        {63, 80, 35, 76, 74, 55, 0, 61, 48, 70, 28, 90, 60, 82, 77},
        {38, 77, 43, 25, 69, 40, 61, 0, 87, 58, 35, 76, 88, 79, 52},
        {93, 50, 61, 84, 57, 79, 48, 87, 0, 46, 41, 54, 38, 49, 75},
        {76, 42, 72, 56, 33, 63, 70, 58, 46, 0, 21, 83, 92, 71, 36},
        {15, 26, 53, 33, 22, 34, 28, 35, 41, 21, 0, 57, 40, 50, 19},
        {48, 73, 88, 66, 80, 58, 90, 76, 54, 83, 57, 0, 62, 40, 64},
        {60, 85, 79, 72, 47, 41, 60, 88, 38, 92, 40, 62, 0, 27, 71},
        {67, 58, 60, 94, 59, 67, 82, 79, 49, 71, 50, 40, 27, 0, 68},
        {44, 30, 69, 41, 73, 29, 77, 52, 75, 36, 19, 64, 71, 68, 0}
    };

    if (n > MAXN) {
        cerr << "n exceeds MAXN\n";
        return 1;
    }

    bitset<MAXN> mask;
    mask.set(0);

    omp_set_num_threads(omp_get_max_threads());
    int maxThreads = omp_get_max_threads();
    parallelLevel = (int)ceil(log2(maxThreads));

    auto t1 = high_resolution_clock::now();
#pragma omp parallel
    {
#pragma omp single nowait
        tsp(0, mask, 0, 0);
    }
    auto t2 = high_resolution_clock::now();

    cout << "Minimum cost Hamiltonian cycle: " << bestCost << "\n";
    cout << "Time taken: "
        << duration_cast<nanoseconds>(t2 - t1).count() / 1e6
        << " ms\n";
    return 0;
}
