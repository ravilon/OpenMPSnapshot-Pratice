#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <omp.h>

using namespace std;

const int NY = 1000;       // Mantemos NY fixo para simplificar
const int STEPS = 500;
const double DX = 1.0;
const double DY = 1.0;
const double DT = 0.6;
const double NU = 0.7;

using Grid = vector<vector<double>>;

void inicializar(Grid& u, int NX) {
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            u[i][j] = 0.0;

    u[NX / 2][NY / 2] = 10.0;
}

void passo_temporal(const Grid& u, Grid& u_new, int NX) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            u_new[i][j] = u[i][j] + NU * DT * (
                (u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]) / (DX * DX) +
                (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) / (DY * DY)
            );
        }
    }
}

void atualizar(Grid& u, const Grid& u_new, int NX) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            u[i][j] = u_new[i][j];
        }
    }
}

int main() {
    int thread_counts[] = {1, 2, 4, 8, 16, 32};
    int base_NX = 100;

    for (int t = 0; t < 6; ++t) {
        int num_threads = thread_counts[t];
        int NX = base_NX * (1 << t);  // Dobra NX a cada iteração: 1000, 2000, 4000, 8000...

        omp_set_num_threads(num_threads);

        Grid u(NX, vector<double>(NY));
        Grid u_new(NX, vector<double>(NY));
        inicializar(u, NX);

        double inicio = omp_get_wtime();

        for (int passo = 0; passo < STEPS; passo++) {
            passo_temporal(u, u_new, NX);
            atualizar(u, u_new, NX);
        }

        double fim = omp_get_wtime();
        cout << "Threads: " << num_threads << " - NX: " << NX << " - Tempo: " << fim - inicio << " s" << endl;
    }

    return 0;
}
