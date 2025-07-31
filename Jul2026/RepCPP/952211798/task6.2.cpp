#include <iostream>
#include <iomanip>  
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

int main() {
    // NÚMERO DE PONTOS A SEREM GERADOS
    const int totalPontos = 10000000;
    int hit = 0;

    // INICIALIZA A SEMENTE DO GERADOR DE NÚMEROS ALEATÓRIOS
    srand(static_cast<unsigned>(time(nullptr)));        

    #pragma omp parallel for
    for (int i = 0; i < totalPontos; ++i) {
        // GERA COORDENADAS (X, Y) ENTRE 0 E 1
        double x = static_cast<double>(rand()) / RAND_MAX;
        double y = static_cast<double>(rand()) / RAND_MAX;

        // VERIFICA SE O PONTO ESTÁ DENTRO DO QUARTO DE CÍRCULO
        if (x * x + y * y <= 1.0) {
            ++hit;
        }
    }

    // ESTIMATIVA DE PI USANDO A RAZÃO DE PONTOS DENTRO DO CÍRCULO
    double pi = 4.0 * hit / totalPontos;

    // EXIBE PI COM 9 CASAS DECIMAIS
    cout << fixed << setprecision(9);
    cout << "Estimativa de pi: " << pi << endl;

    return 0;
}
