#include <iostream>
#include <iomanip>
#include <random>
#include <ctime>
#include <omp.h>

using namespace std;

int main() {
    // NÚMERO TOTAL DE PONTOS A SEREM GERADOS
    const int totalPontos = 1000000000;
    int hit = 0;

    // REGIÃO PARALELA COM ESCOPO EXPLÍCITO DAS VARIÁVEIS (DEFAULT(NONE))
    // COMPARTILHA totalPontos E hit ENTRE AS THREADS
    #pragma omp parallel default(none) shared(totalPontos, hit)
    {
        // CRIADOR DE NÚMEROS ALEATÓRIOS LOCAL PARA CADA THREAD
        std::mt19937 rng(time(nullptr) + omp_get_thread_num());

        // DISTRIBUIÇÃO UNIFORME ENTRE 0.0 E 1.0
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // CONTADOR LOCAL DE PONTOS DENTRO DO CÍRCULO PARA CADA THREAD
        int local_hit = 0;

        // LOOP PARALELIZADO ENTRE AS THREADS
        #pragma omp for
        for (int i = 0; i < totalPontos; ++i) {
            // GERA UM PAR DE COORDENADAS (X, Y)
            double x = dist(rng);
            double y = dist(rng);

            // VERIFICA SE O PONTO ESTÁ DENTRO DO QUARTO DE CÍRCULO
            if (x * x + y * y <= 1.0) {
                ++local_hit;
            }
        }

        // ATUALIZA A VARIÁVEL GLOBAL hit DE FORMA ATÔMICA
        #pragma omp atomic
        hit += local_hit;
    }

    // CALCULA A ESTIMATIVA DE PI COM BASE NA RAZÃO DOS PONTOS
    double pi = 4.0 * hit / totalPontos;

    // EXIBE O RESULTADO COM 9 CASAS DECIMAIS
    cout << fixed << setprecision(9);
    cout << "Estimativa de pi: " << pi << endl;

    return 0;
}
