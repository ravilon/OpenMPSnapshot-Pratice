#include <iostream>     
#include <fstream>      
#include <vector>       
#include <iomanip>      
#include <omp.h>         // BIBLIOTECA PARA PARALELIZAÇÃO COM OPENMP

using namespace std;    

const int NX = 200;     // TAMANHO DO GRID NA DIREÇÃO X
const int NY = 1000;     // TAMANHO DO GRID NA DIREÇÃO Y
const int STEPS = 1000;  // NÚMERO DE PASSOS DE TEMPO
const double DX = 1.0;  // DISTÂNCIA ENTRE PONTOS NA DIREÇÃO X
const double DY = 1.0;  // DISTÂNCIA ENTRE PONTOS NA DIREÇÃO Y
const double DT = 0.6;  // INTERVALO DE TEMPO
const double NU = 0.7;  // COEFICIENTE DE VISCOSIDADE CINEMÁTICA

using Grid = vector<vector<double>>; // DEFINIÇÃO DE UM TIPO GRID COMO MATRIZ DE DOUBLE

// FUNÇÃO PARA INICIALIZAR A VELOCIDADE DO FLUIDO
void inicializar(Grid& u) {
    for (int i = 0; i < NX; i++)            // LOOP SOBRE O EIXO X
        for (int j = 0; j < NY; j++)        // LOOP SOBRE O EIXO Y
            u[i][j] = 0.0;                  // INICIALIZA TODOS OS PONTOS COM ZERO

    u[NX / 2][NY / 2] = 10.0;               // INSERE UMA PERTURBAÇÃO NO CENTRO DO DOMÍNIO
}

// FUNÇÃO PARA SALVAR O GRID EM UM ARQUIVO CSV
void salvar_csv(const Grid& u, int passo) {
    ofstream file("saida_" + to_string(passo) + ".csv");  // CRIA UM ARQUIVO COM NOME BASEADO NO PASSO
    file << fixed << setprecision(4);                     // FORMATA OS NÚMEROS COM 4 CASAS DECIMAIS

    for (int i = 0; i < NX; i++) {                        // LOOP SOBRE O EIXO X
        for (int j = 0; j < NY; j++)                      // LOOP SOBRE O EIXO Y
            file << u[i][j] << (j < NY - 1 ? "," : "");   // ESCREVE OS VALORES SEPARADOS POR VÍRGULA
        file << "\n";                                     // PULA PARA A PRÓXIMA LINHA
    }

    file.close(); // FECHA O ARQUIVO
}

// FUNÇÃO PARA CALCULAR UM PASSO DE TEMPO USANDO A DIFUSÃO (LAPLACIANO)
void passo_temporal(const Grid& u, Grid& u_new) {
    #pragma omp parallel for collapse(2) schedule(static) // PARALELIZAÇÃO COM OPENMP: COLLAPSE DE 2 LAÇOS, SCHEDULE ESTÁTICO
    for (int i = 1; i < NX - 1; i++) {                                        // LOOP SOBRE X, EVITANDO AS BORDAS
        for (int j = 1; j < NY - 1; j++) {                                    // LOOP SOBRE Y, EVITANDO AS BORDAS
            u_new[i][j] = u[i][j] + NU * DT * (
                (u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]) / (DX * DX) +       // SEGUNDA DERIVADA EM X
                (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) / (DY * DY)         // SEGUNDA DERIVADA EM Y
            );
        }
    }
}

// FUNÇÃO PARA COPIAR OS VALORES DE u_new PARA u
void atualizar(Grid& u, const Grid& u_new) {
    #pragma omp parallel for collapse(2) schedule(static) // PARALELIZAÇÃO COM OPENMP: COLLAPSE DE 2 LAÇOS, SCHEDULE ESTÁTICO
    for (int i = 1; i < NX - 1; i++) {           // LOOP SOBRE X, EXCLUINDO AS BORDAS
        for (int j = 1; j < NY - 1; j++) {       // LOOP SOBRE Y, EXCLUINDO AS BORDAS
            u[i][j] = u_new[i][j];               // ATUALIZA O VALOR DE u COM O NOVO CÁLCULO
        }
    }
}

// FUNÇÃO PRINCIPAL
int main() {
    Grid u(NX, vector<double>(NY));              // CRIA A MATRIZ DE VELOCIDADE ORIGINAL
    Grid u_new(NX, vector<double>(NY));          // CRIA A MATRIZ PARA ARMAZENAR O NOVO ESTADO

    inicializar(u);                              // INICIALIZA A MATRIZ COM A PERTURBAÇÃO

    double inicio = omp_get_wtime();             // MARCA O TEMPO INICIAL

    for (int passo = 0; passo < STEPS; passo++) { // LOOP SOBRE OS PASSOS DE TEMPO
        passo_temporal(u, u_new);                // CALCULA A NOVA VELOCIDADE
        atualizar(u, u_new);                     // COPIA O NOVO VALOR PARA A MATRIZ ORIGINAL
    }

    double fim = omp_get_wtime();                               // MARCA O TEMPO FINAL
    cout << "Tempo total (serial): " << fim - inicio << " s\n"; // EXIBE O TEMPO GASTO

    return 0; 
}
