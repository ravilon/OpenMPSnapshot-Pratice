#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

using namespace std;

static string schedulingMethod;
static int chunkSize;
static int M;
static int executionTime;
static int t;

// Calculates prime numbers in range [start, end].
int primeNumberGenerator(vector<int> &primesIn, vector<int> &primesOut, int start, int end) {
    int realStart = start;
    if (start <= 2) {
        primesOut.push_back(2);
        realStart = 1;
    } else if (start % 2 == 0) {
        realStart -= 1;
    } else {
        realStart -= 2;
    }
    int j;
    int k;
    int n;
    int quo, rem;
    n = realStart;

    for (j = 1; n <= end; j++) {// P2
        if (j != 1) primesOut.push_back(n);
        bool check = true;
        while (check) {//p4
            n += 2;
            for (k = 1; k < primesIn.size(); k++) {//p6
                quo = n / primesIn.at(k);
                rem = n % primesIn.at(k);
                if (rem == 0) {
                    break;
                }
                if (quo <= primesIn.at(k)) {
                    check = false;
                    break;
                }
            }
            if (k >= primesIn.size()) check = false;
        }

    }
}

// Calculates prime numbers up to end.
int primeNumberGenerator(vector<int> &primes, int end) {
    primeNumberGenerator(primes, primes, 1, end);
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cout << "All arguments are required: Scheduling method, chunk size, M, number of threads.";
        return 0;
    }
    time_t begin_time;
    time(&begin_time);

    schedulingMethod = argv[1];
    transform(schedulingMethod.begin(), schedulingMethod.end(), schedulingMethod.begin(), ::toupper);
    chunkSize = stoi(argv[2]);
    M = stoi(argv[3]);
    t = stoi(argv[4]);
    static bool print = false;
    try {
        string option = argv[5];
        if (option.compare( "--print") == 0) print = true;
    } catch (exception e) {
    }


    vector<int> prime;
    int squareRootM = (int) sqrt(M);
    primeNumberGenerator(prime, squareRootM);

    omp_set_num_threads(t);
    /*
     * For the schedule kinds static, dynamic, and guided the chunk_size is set to the value of the second argument,
     * or to the default chunk_size if the value of the second argument is less than 1; for the schedule kind auto the
     * second argument has no meaning; for implementation specific schedule kinds, the values and associated meanings
     * of the second argument are implementation defined. */
    if (schedulingMethod.compare("STATIC")) {
        omp_set_schedule(omp_sched_static, chunkSize);
    } else if (schedulingMethod.compare("DYNAMIC")) {
        omp_set_schedule(omp_sched_dynamic, chunkSize);
    } else if (schedulingMethod.compare("GUIDED")) {
        omp_set_schedule(omp_sched_guided, chunkSize);
    } else if (schedulingMethod.compare("AUTO")) {
        omp_set_schedule(omp_sched_auto, chunkSize);
    }

    vector<vector<int>> threads_prime;
    #pragma omp parallel shared(threads_prime, M, squareRootM, prime)
    {
        vector<int> primesOut;

        #pragma omp for schedule(runtime) nowait
        for (int i = squareRootM + 1; i <= M; i += 2) {
            primeNumberGenerator(prime, primesOut, i, i + 1);
        }
        #pragma omp critical
        threads_prime.push_back(primesOut);
    }

    for (auto tmp : threads_prime) {
        for (int j : tmp) {
            prime.push_back(j);
        }
    }

    time_t end_time;
    time(&end_time);
    executionTime = end_time - begin_time;

    sort(prime.begin(), prime.end());

    if (print) {
        for (int j : prime)
            printf("%d\n", j);
    }

    printf("Execution time: %d\n", executionTime);
    return 0;
}
