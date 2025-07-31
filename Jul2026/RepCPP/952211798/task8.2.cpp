#include <iostream>      
#include <omp.h>         
#include <cstdlib>       
#include <ctime>          
#include <vector>

using namespace std;

int main() {
    int N = 100000000;   // TOTAL DE PONTOS A SEREM GERADOS
    int total_in = 0;    // CONTADOR GLOBAL DE PONTOS DENTRO DO CÍRCULO
    int num_threads;     // VARIÁVEL PARA ARMAZENAR O NÚMERO DE THREADS
    vector<int> acertos; // VETOR PARA ARMAZENAR ACERTOS DE CADA THREAD

    double start = omp_get_wtime();  // INÍCIO DA MEDIÇÃO DE TEMPO

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();  // ID DA THREAD

        #pragma omp single  // GARANTE QUE APENAS UMA THREAD EXECUTE ESTE BLOCO
        {
            num_threads = omp_get_num_threads();  // OBTÉM O NÚMERO DE THREADS
            acertos.resize(num_threads, 0);       // REDIMENSIONA O VETOR PARA O NÚMERO DE THREADS
        }

        #pragma omp for  // DISTRIBUI O LAÇO ENTRE AS THREADS
        for (int i = 0; i < N; i++) {
            float x = (float)rand() / RAND_MAX;  // GERA UM X ALEATÓRIO ENTRE 0 E 1
            float y = (float)rand() / RAND_MAX;  // GERA UM Y ALEATÓRIO ENTRE 0 E 1
            if (x * x + y * y <= 1.0f)           // VERIFICA SE O PONTO ESTÁ DENTRO DO CÍRCULO
                acertos[tid]++;                  // INCREMENTA O CONTADOR LOCAL DA THREAD
        }
    }

    for (int i = 0; i < num_threads; i++) {
        total_in += acertos[i];  // SOMA OS ACERTOS DE TODAS AS THREADS
    }

    double pi = 4.0 * total_in / N;  // CALCULA A ESTIMATIVA DE PI
    double end = omp_get_wtime();    // FIM DA MEDIÇÃO DE TEMPO

    cout << "[rand + vetor] PI = " << pi << endl;                    // EXIBE O RESULTADO DE PI
    cout << "[rand + vetor] Tempo: " << end - start << " segundos" << endl;  // EXIBE O TEMPO DE EXECUÇÃO
    return 0;
}
