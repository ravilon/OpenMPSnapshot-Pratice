#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <iomanip>

using namespace std;
using namespace chrono;

bool ehPrimo(int n) {
    int raiz = sqrt(n);
    if (pow(raiz, 2) == n) { return false; }

    for (int i = 2; i <= raiz; ++i) {
        if (n % i == 0) { return false; }
    }
    return true;
}

int main() {
    int nPrimos = 0; int n; 
    vector<int> testes = {5000, 10000, 50000, 100000, 500000, 1000000, 2000000, 3000000, 4000000, 5000000};

    for (int i=0; i < testes.size(); i++) {
        n = testes[i];

        auto start = high_resolution_clock::now();
        #pragma omp parallel for reduction(+:nPrimos)
        for (int i = 0; i < n; i++) {
            if (ehPrimo(i + 2)) {
                nPrimos += 1;
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start).count();
        double seconds = duration / 1e9;

        cout << fixed << setprecision(9);
        cout << "NUMERO: "<< n << " RESULTADO: " << nPrimos << " " << "TEMPO: " << seconds << "s" << endl;
    }

    return 0;
}
