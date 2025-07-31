#include <iostream>
#include <random>
#include <omp.h>
#include <ctime>

using namespace std;

int main() {
    int N = 100000000;              // NÚMERO DE PONTOS
    int total_in = 0;               // TOTAL DE PONTOS DENTRO DO CÍRCULO
    double start = omp_get_wtime(); // INÍCIO DO TEMPO DE EXECUÇÃO

    #pragma omp parallel            // INICIA A REGIÃO PARALELA
    {
        std::mt19937 rng(time(nullptr) + omp_get_thread_num()); // GERADOR INDEPENDENTE POR THREAD
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);   // DISTRIBUIÇÃO UNIFORME

        #pragma omp for reduction(+:total_in)   // DISTRIBUI O LOOP E REDUZ O CONTADOR GLOBAL
        for (int i = 0; i < N; i++) {
            float x = dist(rng);      // GERA NÚMERO ALEATÓRIO PARA X
            float y = dist(rng);      // GERA NÚMERO ALEATÓRIO PARA Y
            if (x * x + y * y <= 1.0f) { // VERIFICA SE O PONTO ESTÁ NO CÍRCULO
                total_in++;          // INCREMENTA O CONTADOR LOCAL DE ACERTOS
            }
        }
    }

    double pi = 4.0 * total_in / N;   // CALCULA O VALOR DE PI
    double end = omp_get_wtime();     // FINALIZA O TEMPO DE EXECUÇÃO

    cout << "[reduction + <random>] PI = " << pi << endl; // EXIBE O VALOR DE PI
    cout << "[reduction + <random>] Tempo: " << end - start << " segundos" << endl; // EXIBE O TEMPO GASTO
    return 0;
}
