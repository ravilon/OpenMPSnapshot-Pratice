#include <iostream>
#include <vector>
#include <omp.h>
#include <valarray>

using namespace std;

int main() {

    int left, right;
    printf("Enter borders: \n");
    cin >> left >> right;

    vector<int> result;

    #pragma omp parallel for
    for (int i = left; i <= right; i++) {
        bool isSimple = true;

        #pragma omp parallel for
        for (int j = 2; j <= (int) sqrt(i); j++) {
            if (i % j == 0) {
                isSimple = false;
            }
        }

        #pragma omp critical
        if (isSimple) {
            result.push_back(i);
        }  
    }

    sort(result.begin(), result.end());

    printf("Simple numbers in [%d, %d]: \n", left, right);
    for (int x: result) {
        printf("%d ", x);
    }
    printf("\n");

    return 0;
}