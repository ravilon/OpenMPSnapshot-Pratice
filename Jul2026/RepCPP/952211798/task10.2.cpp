#include <iostream>
#include <random>
#include <omp.h>
#include <ctime>

using namespace std;

int main() {
    int N = 100000000;              // NÚMERO DE PONTOS
    int total_in = 0;               // TOTAL DE PONTOS DENTRO DO CÍRCULO
    double start = omp_get_wtime(); // MARCA O INÍCIO DO TEMPO

    #pragma omp parallel            // INICIA A REGIÃO PARALELA
    {
        mt19937 rng(time(nullptr) + omp_get_thread_num());   // GERADOR DE NÚMEROS POR THREAD
        uniform_real_distribution<float> dist(0.0f, 1.0f);   // DISTRIBUIÇÃO UNIFORME

        #pragma omp for               // DISTRIBUI O LOOP ENTRE AS THREADS
        for (int i = 0; i < N; i++) {
            float x = dist(rng);      // GERA NÚMERO ALEATÓRIO PARA X
            float y = dist(rng);      // GERA NÚMERO ALEATÓRIO PARA Y
            if (x * x + y * y <= 1.0f) { // VERIFICA SE O PONTO ESTÁ NO CÍRCULO
                #pragma omp atomic     // GARANTE ATUALIZAÇÃO SEGURA DA VARIÁVEL GLOBAL
                total_in++;          // INCREMENTA O ACERTO GLOBAL
            }
        }
    }

    double pi = 4.0 * total_in / N;   // CALCULA O VALOR DE PI
    double end = omp_get_wtime();     // MARCA O FIM DO TEMPO

    cout << "[atomic + <random>] PI = " << pi << endl; 
    cout << "[atomic + <random>] Tempo: " << end - start << " segundos" << endl; 
    return 0;
}
