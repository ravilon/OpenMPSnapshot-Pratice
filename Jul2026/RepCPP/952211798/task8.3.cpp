#include <iostream>      
#include <omp.h>         
#include <random>        
#include <ctime>         

using namespace std;

int main() {
    int N = 100000000;   // TOTAL DE PONTOS A SEREM GERADOS
    int total_in = 0;    // CONTADOR GLOBAL DE PONTOS DENTRO DO CÍRCULO
    double start = omp_get_wtime();  // INÍCIO DA MEDIÇÃO DE TEMPO

    #pragma omp parallel
    {
        int local_in = 0;  // CONTADOR LOCAL PARA CADA THREAD
        unsigned int tid = omp_get_thread_num();  // ID DA THREAD
        std::minstd_rand rng(time(nullptr) + tid);  // INICIALIZA O GERADOR DE NÚMEROS ALEATÓRIOS PARA CADA THREAD
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // DISTRIBUIÇÃO UNIFORME DE PONTOS ALEATÓRIOS

        #pragma omp for  // DISTRIBUI O LAÇO ENTRE AS THREADS
        for (int i = 0; i < N; i++) {
            float x = dist(rng);  // GERA UM X ALEATÓRIO ENTRE 0 E 1
            float y = dist(rng);  // GERA UM Y ALEATÓRIO ENTRE 0 E 1
            if (x * x + y * y <= 1.0f)  // VERIFICA SE O PONTO ESTÁ DENTRO DO CÍRCULO
                local_in++;  // INCREMENTA O CONTADOR LOCAL
        }

        #pragma omp critical  // SEÇÃO CRÍTICA PARA ATUALIZAR O CONTADOR GLOBAL
        {
            total_in += local_in;  // ACUMULA O RESULTADO DA THREAD NO TOTAL GLOBAL
        }
    }

    double pi = 4.0 * total_in / N;  // CALCULA A ESTIMATIVA DE PI
    double end = omp_get_wtime();    // FIM DA MEDIÇÃO DE TEMPO

    cout << "[minstd_rand + critical] PI = " << pi << endl;  // EXIBE O RESULTADO DE PI
    cout << "[minstd_rand + critical] Tempo: " << end - start << " segundos" << endl;  // EXIBE O TEMPO DE EXECUÇÃO
    return 0;
}
