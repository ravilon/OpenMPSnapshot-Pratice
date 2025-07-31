#include <iostream>
#include <list>
#include <vector>
#include <random>
#include <omp.h>

using namespace std;

// LISTAS GLOBAIS PARA A PRIMEIRA PARTE
list<int> listaA;
list<int> listaB;

// FUNÇÃO PARA GERAR UM VALOR ALEATÓRIO
int gerarValor() {
    static thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> distrib(1, 1000);
    return distrib(gen);
}

// PARTE 1: DUAS LISTAS COM REGIÕES CRÍTICAS NOMEADAS
void insercaoComRegioesCriticas(int N) {

    #pragma omp parallel num_threads(2)
    {
        int id = omp_get_thread_num(); // OBTÉM O NÚMERO DA THREAD
        for (int i = 0; i < N; ++i) {
            int valor = gerarValor(); // GERA UM VALOR ALEATÓRIO
            int destino = valor % 2;  // SIMULA ESCOLHA RANDOM DE 0 OU 1

            if (destino == 0) {
                #pragma omp critical(listaA) // BLOQUEIA O ACESSO À LISTA A
                listaA.push_back(valor); // INSERE O VALOR NA LISTA A
            } else {
                #pragma omp critical(listaB) // BLOQUEIA O ACESSO À LISTA B
                listaB.push_back(valor); // INSERE O VALOR NA LISTA B
            }
        }
    }

    cout << "\n[PARTE 1] INSERCAO COM 2 LISTAS FIXAS\n";
    cout << "TAMANHO DA LISTA A: " << listaA.size() << "\n"; // EXIBE O TAMANHO DA LISTA A
    cout << "TAMANHO DA LISTA B: " << listaB.size() << "\n"; // EXIBE O TAMANHO DA LISTA B
}

// PARTE 2: GENERALIZAÇÃO COM N LISTAS USANDO LOCKS EXPLÍCITOS
void insercaoGeneralizada(int numListas, int N, int numThreads) {
    
    vector<list<int>> listas(numListas); // CRIA VETOR DE LISTAS
    vector<omp_lock_t> locks(numListas); // CRIA VETOR DE LOCKS

    // INICIALIZA OS LOCKS
    for (int i = 0; i < numListas; ++i) {
        omp_init_lock(&locks[i]); // INICIALIZA O LOCK PARA CADA LISTA
    }

    #pragma omp parallel num_threads(numThreads)
    {
        int id = omp_get_thread_num(); // OBTÉM O NÚMERO DA THREAD
        mt19937 gen(random_device{}() + id); // GERADOR DE NÚMEROS ALEATÓRIOS POR THREAD
        uniform_int_distribution<> distLista(0, numListas - 1); // ÍNDICE ALEATÓRIO DE LISTA
        uniform_int_distribution<> distValor(1, 1000); // VALOR CONVENCIONAL

        for (int i = 0; i < N; ++i) {
            int indice = distLista(gen); // DEFINE LISTA RANDOM
            int valor = distValor(gen);  // DEFINE VALOR CONVENCIONAL

            omp_set_lock(&locks[indice]);   // BLOQUEIA O LOCK DA LISTA
            listas[indice].push_back(valor);// INSERE VALOR NA LISTA
            omp_unset_lock(&locks[indice]); // LIBERA O LOCK
        }
    }

    cout << "\n[PARTE 2] INSERCAO GENERALIZADA COM " << numListas << " LISTAS\n";
    for (int i = 0; i < numListas; ++i) {
        cout << "TAMANHO DA LISTA " << i << ": " << listas[i].size() << "\n"; // EXIBE O TAMANHO DE CADA LISTA
    }

    // LIBERA LOCKS
    for (int i = 0; i < numListas; ++i) {
        omp_destroy_lock(&locks[i]); // LIBERA OS LOCKS AO FINAL
    }
}

int main() {
    int N = 500000;

    // PARTE 1: DUAS LISTAS COM REGIÕES CRÍTICAS NOMEADAS
    insercaoComRegioesCriticas(N);

    // PARTE 2: GENERALIZAÇÃO PARA MÚLTIPLAS LISTAS
    int numListas = 4;
    int numThreads = 4;
    insercaoGeneralizada(numListas, N, numThreads);

    return 0;
}
