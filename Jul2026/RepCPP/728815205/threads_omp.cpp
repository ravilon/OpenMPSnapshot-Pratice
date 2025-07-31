// Projeto de Supercomputação | Engenharia de Computação - Insper
// Professores: André Filipe e Michel Fornaciali

#include <iostream>  // Inclui a biblioteca padrão de entrada e saída.
#include <vector>    // Inclui a biblioteca para usar vetores dinâmicos.
#include <algorithm> // Inclui a biblioteca para operações de algoritmo como sort.
#include <omp.h>     // Inclui a biblioteca OpenMP para paralelização.
#include <fstream>   // Inclui a biblioteca para manipulação de arquivos.
#include <chrono>    // Inclui a biblioteca para medir o tempo.

using namespace std;

// Função para ler um grafo de um arquivo e retornar sua matriz de adjacência.
vector<vector<int>> LerGrafo(const string& fileName, int& qntVertices) {
    ifstream arquivo(fileName);  // Abre o arquivo para leitura.
    int numArestas;
    arquivo >> qntVertices >> numArestas;  // Lê a quantidade de vértices e arestas.

    vector<vector<int>> grafo(qntVertices, vector<int>(qntVertices, 0));  // Cria matriz de adjacência.

    for (int i = 0; i < numArestas; ++i) {
        int u, v;
        arquivo >> u >> v;  // Lê as arestas.
        grafo[u-1][v-1] = 1;  // Preenche a matriz de adjacência.
        grafo[v-1][u-1] = 1;
    }

    arquivo.close();  // Fecha o arquivo.
    return grafo;  // Retorna a matriz de adjacência.
}

vector<vector<int>> cliquesMaximasGlobal;  // Vetor global para armazenar cliques maximas.
vector<int> cliqueMaximaGlobal;  // Vetor para armazenar a maior clique máxima.

// Função que utiliza OpenMP para encontrar cliques maximas.
void cliquesMaximasOpenMP(const vector<vector<int>>& grafo, int qntVertices) {
    #pragma omp parallel  // Inicia uma região paralela OpenMP.
    {
        vector<bool> visitados(qntVertices, false);  // Vetor de visitados.
        vector<int> cliqueAtual;  // Vetor para armazenar a clique atual.

        #pragma omp for nowait  // Paraleliza o loop com OpenMP.
        for (int i = 0; i < qntVertices; ++i) {
            // Chama a função recursiva para cada vértice.
            encontrarCliqueMaximoRecursivo(grafo, cliqueAtual, visitados, i);
        }
    }

    // Ordena e remove duplicatas das cliques maximas.
    sort(cliquesMaximasGlobal.begin(), cliquesMaximasGlobal.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size();
    });

    cliquesMaximasGlobal.erase(unique(cliquesMaximasGlobal.begin(), cliquesMaximasGlobal.end()), cliquesMaximasGlobal.end());
}

// Função para verificar se um nó pode ser adicionado a uma clique.
bool adicionarNoClique(const vector<vector<int>>& grafo, const vector<int>& cliqueAtual, int vizinho) {
    for (auto j : cliqueAtual) {
        if (grafo[vizinho][j] == 0) {
            return false;
        }
    }
    return true;
}

// Função para encontrar todos os possíveis vizinhos para a clique atual.
void possiveisVizinhos(const vector<vector<int>>& grafo, vector<int>& possiveisVertices, const vector<int>& cliqueAtual, const vector<bool>& visitados) {
    for (int i = 0; i < grafo.size(); ++i) {
        if (!visitados[i] && adicionarNoClique(grafo, cliqueAtual, i)) {
            possiveisVertices.emplace_back(i);
        }
    }
}

// Função para atualizar o vetor global de cliques maximas.
void atualizarCliques(const vector<int>& cliqueAtual) {
    #pragma omp critical  // Região crítica para garantir a integridade dos dados.
    {
        cliquesMaximasGlobal.push_back(cliqueAtual);
        if (cliqueAtual.size() > cliqueMaximaGlobal.size()) {
            cliqueMaximaGlobal = cliqueAtual;
        }
    }
}

// Função recursiva para encontrar a clique máxima.
void encontrarCliqueMaximoRecursivo(const vector<vector<int>>& grafo, vector<int>& cliqueAtual, vector<bool>& visitados, int vertice) {
    cliqueAtual.push_back(vertice);  // Adiciona o vértice à clique atual.
    visitados[vertice] = true;  // Marca o vértice como visitado.

    vector<int> possiveisVertices;  // Vetor para armazenar os possíveis vizinhos.
    possiveisVizinhos(grafo, possiveisVertices, cliqueAtual, visitados);  // Encontra possíveis vizinhos.

    for (auto vizinho : possiveisVertices) {
        // Chamada recursiva para cada vizinho possível.
        encontrarCliqueMaximoRecursivo(grafo, cliqueAtual, visitados, vizinho);
    }

    if (possiveisVertices.empty() && cliqueAtual.size() > 1) {
        atualizarCliques(cliqueAtual);  // Atualiza as cliques maximas.
    }

    cliqueAtual.pop_back();  // Remove o último vértice da clique atual.
    visitados[vertice] = false;  // Marca o vértice como não visitado.
}

int main(int argc, char* argv[]) {
    // Verifica se o nome do arquivo foi passado como argumento.
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <nome_do_arquivo_de_entrada>" << endl;
        return 1;
    }

    string nome_arquivo_entrada = argv[1];  // Obtém o nome do arquivo de entrada.
    int qntVertices;
    vector<vector<int>> grafo = LerGrafo(nome_arquivo_entrada, qntVertices);  // Lê o grafo do arquivo.

    auto start_time = chrono::high_resolution_clock::now();  // Inicia a contagem do tempo.
    cliquesMaximasOpenMP(grafo, qntVertices);  // Encontra cliques maximas com OpenMP.
    
    // Encerra a contagem do tempo.
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    // Imprime as cliques maximas encontradas.
    for (const auto& clique : cliquesMaximasGlobal) {
        cout << "[";
        for (int i = 0; i < clique.size(); ++i) {
            cout << clique[i] + 1;
            if (i < clique.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }

    // Imprime a maior clique máxima encontrada.
    cout << "Clique máxima encontrada (OpenMP): [";
    for (int i = 0; i < cliqueMaximaGlobal.size(); ++i) {
        cout << cliqueMaximaGlobal[i] + 1;
        if (i < cliqueMaximaGlobal.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    cout << "Tempo de execução: " << elapsed.count() << "s" << endl;

    return 0;
}
