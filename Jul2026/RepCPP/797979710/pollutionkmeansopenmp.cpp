/*
Alunos: Rafael Ferreira Garcia Almeida, Samuel Spineli Rodrigues, Lucas Fiori Magalhaes Machado

Tempo Sequencial : 34.4797 segundos

Apenas com o OpenMP:
Tempo Paralelo 1 thread : 50.4282 segundos
Tempo Paralelo 2 threads : 18.4068 segundos
Tempo Paralelo 4 threads : 10.5695 segundos
Tempo Paralelo 8 threads : 12.2925 segundos

Apenas com MPI
1 processo: 12.7478 segundos
2 processos: 6.6243 segundos
4 processos: 3.7032 segundos
8 processos: 14.1557 segundos

Com OpenMP e MPI
1 processo 4 threads : 9.8449 segundos
2 processos 2 threads : 3.6188 segundos
4 processos 0 threads : 3.7032 segundos
8 processos 0 threads : 14.1557 segundos
*/



#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono> // Para medir o tempo de execução
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Definindo uma estrutura para representar uma amostra
struct Sample {
    double value;
    double latitude;
    double longitude;
};

// Função para calcular a distância euclidiana entre duas amostras
inline double euclideanDistance(const Sample& sample1, const Sample& sample2) {
    return sqrt(pow(sample1.value - sample2.value, 2) +
                pow(sample1.latitude - sample2.latitude, 2) +
                pow(sample1.longitude - sample2.longitude, 2));
}

// Função para inicializar os centróides dos clusters aleatoriamente
vector<Sample> initializeCentroids(const vector<Sample>& samples, int numClusters) {
    vector<Sample> centroids;
    vector<int> centroidIndices;
    int numSamples = samples.size();

    // Gerar índices únicos aleatórios para os centróides
    while (centroidIndices.size() < numClusters) {
        int index = rand() % numSamples;
        if (std::find(centroidIndices.begin(), centroidIndices.end(), index) == centroidIndices.end()) {
            centroidIndices.push_back(index);
        }
    }

    // Inicializar os centróides com as amostras correspondentes
    for (int i = 0; i < numClusters; ++i) {
        centroids.push_back(samples[centroidIndices[i]]);
    }

    return centroids;
}

// Função para atribuir amostras aos clusters mais próximos
void assignSamplesToClusters(const vector<Sample>& samples, const vector<Sample>& centroids, vector<int>& labels) {
    int numSamples = samples.size();
    int numClusters = centroids.size();

    #pragma omp parallel for // 
    for (int i = 0; i < numSamples; ++i) {
        double minDistance = numeric_limits<double>::max();
        int closestCluster = 0;
        for (int j = 0; j < numClusters; ++j) {
            double distance = euclideanDistance(samples[i], centroids[j]);
            if (distance < minDistance) {
                minDistance = distance;
                closestCluster = j;
            }
        }
        labels[i] = closestCluster;
    }
}

// Função para recalcular os centróides dos clusters
void updateCentroids(const vector<Sample>& samples, const vector<int>& labels, vector<Sample>& centroids) {
    int numClusters = centroids.size();
    int numFeatures = 3; // value, latitude, longitude

    vector<int> clusterCounts(numClusters, 0);
    vector<double> sumValues(numClusters, 0);
    vector<double> sumLatitudes(numClusters, 0);
    vector<double> sumLongitudes(numClusters, 0);

    // Calcular a soma dos valores, latitudes e longitudes para cada cluster
    #pragma omp parallel for
    for (int i = 0; i < samples.size(); ++i) {
        int cluster = labels[i];
        sumValues[cluster] += samples[i].value;
        sumLatitudes[cluster] += samples[i].latitude;
        sumLongitudes[cluster] += samples[i].longitude;
        clusterCounts[cluster]++;
    }

    // Atualizar os centróides dividindo as somas pelos números de amostras em cada cluster
    #pragma omp parallel for
    for (int i = 0; i < numClusters; ++i) {
        if (clusterCounts[i] > 0) {
            centroids[i].value = sumValues[i] / clusterCounts[i];
            centroids[i].latitude = sumLatitudes[i] / clusterCounts[i];
            centroids[i].longitude = sumLongitudes[i] / clusterCounts[i];
        }
    }
}

// Função para executar o algoritmo K-means
vector<int> kmeans(const vector<Sample>& samples, int numClusters, int maxIterations) {
    vector<Sample> centroids = initializeCentroids(samples, numClusters);
    vector<int> labels(samples.size());

    int iteration = 0;

    while (iteration < maxIterations) {
        assignSamplesToClusters(samples, centroids, labels);
        updateCentroids(samples, labels, centroids);
        iteration++;
    }

    return labels;
}

// Função para ler os dados do arquivo CSV e armazená-los em um vetor de amostras
vector<Sample> readCSV(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo " << filename << endl;
        exit(EXIT_FAILURE);
    }

    vector<Sample> samples;
    string line;

    // Ignorar o cabeçalho
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        Sample sample;

        // Ignorar as primeiras 5 colunas
        for (int i = 0; i < 5; ++i) {
            getline(ss, token, ',');
        }

        try {
            // Ler as coordenadas e o valor
            getline(ss, token, ',');
            sample.latitude = stod(token);
            getline(ss, token, ',');
            sample.longitude = stod(token);
            getline(ss, token, ',');
            sample.value = stod(token);

            samples.push_back(sample);
        } catch (const std::invalid_argument& e) {
            cerr << "Erro ao converter string para double: " << e.what() << endl;
            cerr << "Linha mal formatada: " << line << endl;
            // Pular esta linha e continuar
        }
    }

    file.close();

    return samples;
}

int main() {
    int numThreads = 1;
    omp_set_num_threads(numThreads);
    
    string inputFile = "openaq.csv";
    int numClusters = 50; // Alterado para 5 clusters
    int maxIterations = 300; // Alterado para 200 iterações

    // Iniciar o timer
    auto startTime = high_resolution_clock::now();

    // Ler os dados do arquivo CSV
    vector<Sample> samples = readCSV(inputFile);

    // Executar o algoritmo K-means
    vector<int> labels = kmeans(samples, numClusters, maxIterations); // Adicionado maxIterations

    // Parar o timer e calcular o tempo de execução em segundos
    auto stopTime = high_resolution_clock::now();
    auto elapsedTime = duration_cast<duration<double>>(stopTime - startTime);

    // Imprimir os rótulos dos clusters
    for (int i = 0; i < samples.size(); ++i) {
        cout << "Amostra " << i << " pertence ao cluster " << labels[i] << endl;
    }

    // Imprimir o tempo de execução
    cout << "Tempo de execução: " << elapsedTime.count() << " segundos" << endl;

    return 0;
}
