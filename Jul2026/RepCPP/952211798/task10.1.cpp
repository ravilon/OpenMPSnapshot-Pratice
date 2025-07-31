#include <iostream>                      
#include <random>                        
#include <omp.h>                         
#include <ctime>                         

using namespace std;                     

int main() {
    int N = 100000000;                    // NÚMERO TOTAL DE PONTOS
    int total_in = 0;                     // CONTADOR DE ACERTOS NO CÍRCULO
    double start = omp_get_wtime();       // MARCA O TEMPO INICIAL

    #pragma omp parallel                  // INICIA REGIÃO PARALELA
    {
        int contagem_acertos = 0;                            // CONTAGEM LOCAL DE ACERTOS
        mt19937 rng(time(nullptr) + omp_get_thread_num());   // GERADOR DE NÚMEROS POR THREAD
        uniform_real_distribution<float> dist(0.0f, 1.0f);   // DISTRIBUIÇÃO UNIFORME

        #pragma omp for                   // DISTRIBUI O LOOP ENTRE AS THREADS
        for (int i = 0; i < N; i++) {
            float x = dist(rng);          // GERA NÚMERO ALEATÓRIO PARA X
            float y = dist(rng);          // GERA NÚMERO ALEATÓRIO PARA Y
            if (x * x + y * y <= 1.0f)    // VERIFICA SE ESTÁ NO CÍRCULO
                contagem_acertos++;                     // CONTA PONTO DENTRO DO CÍRCULO
        }

        #pragma omp critical                            // ACESSO EXCLUSIVO À VARIÁVEL GLOBAL
        {
            total_in += contagem_acertos;               // ATUALIZA O CONTADOR GLOBAL
        }
    }

    double pi = 4.0 * total_in / N;       // ESTIMA PI
    double end = omp_get_wtime();         // MARCA O TEMPO FINAL

    cout << "[critical + <random>] PI = " << pi << endl;                 
    cout << "[critical + <random>] Tempo: " << end - start << " segundos" << endl; 
    return 0;                              
}
