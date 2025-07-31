#include <iostream>      
#include <omp.h>         
#include <cstdlib>       
#include <ctime>         

using namespace std;     

int main() {
    int N = 100000000;   // TOTAL DE PONTOS A SEREM GERADOS
    int hit = 0;    // CONTADOR GLOBAL DE PONTOS DENTRO DO CÍRCULO
    double start = omp_get_wtime();  // INÍCIO DA MEDIÇÃO DE TEMPO

    #pragma omp parallel   // INÍCIO DA REGIÃO PARALELA
    {
        int local_in = 0;  // CONTADOR LOCAL PARA CADA THREAD

        #pragma omp for    // DISTRIBUIÇÃO DO LAÇO ENTRE AS THREADS
        for (int i = 0; i < N; i++) {
            float x = (float)rand() / RAND_MAX;  // GERA UM X ALEATÓRIO ENTRE 0 E 1
            float y = (float)rand() / RAND_MAX;  // GERA UM Y ALEATÓRIO ENTRE 0 E 1
            if (x * x + y * y <= 1.0f)           // VERIFICA SE O PONTO ESTÁ DENTRO DO CÍRCULO
                hit++;                      // INCREMENTA O CONTADOR LOCAL
        }

        #pragma omp critical       // SEÇÃO CRÍTICA PARA ATUALIZAR O CONTADOR GLOBAL
        {
            hit += local_in;  // ACUMULA O RESULTADO DA THREAD NO TOTAL GLOBAL
        }
    }

    double pi = 4.0 * hit/ N;     // ESTIMATIVA FINAL DE PI
    double end = omp_get_wtime();       // FIM DA MEDIÇÃO DE TEMPO

    cout << "[RAND + CRITICAL] PI = " << pi << endl;                   
    cout << "[RAND + CRITICAL] TEMPO: " << end - start << " segundos" << endl;  
    return 0;   
}
