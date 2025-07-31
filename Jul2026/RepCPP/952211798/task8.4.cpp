#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include <ctime>

using namespace std;

int main() {
    int N = 100000000;  // Total de pontos a serem gerados
    int total_in = 0;   // Contador global
    int num_threads;    // Número total de threads
    vector<int> acertos;

    double start = omp_get_wtime();  // Início da medição de tempo

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // Inicializa gerador de números aleatórios exclusivo por thread
        std::minstd_rand rng(time(nullptr) + tid);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            acertos.resize(num_threads, 0);
        }

        #pragma omp for
        for (int i = 0; i < N; i++) {
            float x = dist(rng);
            float y = dist(rng);
            if (x * x + y * y <= 1.0f)
                acertos[tid]++;
        }
    }

    // Acúmulo serial após a região paralela
    for (int i = 0; i < num_threads; i++) {
        total_in += acertos[i];
    }

    double pi = 4.0 * total_in / N;
    double end = omp_get_wtime();

    cout << "[minstd_rand + vetor] PI = " << pi << endl;
    cout << "[minstd_rand + vetor] Tempo: " << end - start << " segundos" << endl;

    return 0;
}
