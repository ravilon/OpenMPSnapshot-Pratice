#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif					// Za uklanjanje upozorenja iz Visual Studio

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int isPrime(int num);

// gcc -O3 -fopenmp -o program program.c
int main(int argc, char** argv) {

    const char* inputFileName = (char*)calloc(30, sizeof(char));
    const char* outputFileName = (char*)calloc(30, sizeof(char));

    inputFileName = argv[1];
    outputFileName = argv[2];

    if (argc != 3) { inputFileName = "input_0-1000.bin"; outputFileName = "output.txt"; }
    // Ucitavanje ulaznog i izlaznog fajla


    int first = 0, last = 0;
    int numOfRanges = 0, alltogetherPrimes = 0;

    FILE* infp = fopen(inputFileName, "rb");
    FILE* outfp = fopen(outputFileName, "w");

    fread(&numOfRanges, sizeof(int), 1, infp);

    for (int i = 0; i < numOfRanges; i++)
    {
        int count = 0;
        fread(&first, sizeof(int), 1, infp);
        fread(&last, sizeof(int), 1, infp);

        // Brojanje prostih brojeva
        //The #pragma omp parallel for directive tells the compiler to parallelize the loop. 
        //The reduction(+:count) clause tells OpenMP to sum up the local count variables of each thread into a global count variable.
        #pragma omp parallel for reduction(+:count)
        for (int i = first; i <= last; i++) {
            if (isPrime(i)) {
                count++;
            }
        }
        alltogetherPrimes += count;

        // Upis u izlazni fajl
        fprintf(outfp, "The number of prime numbers between %d and %d is: %d\n", first, last, count);
    }

    fprintf(outfp, "\n     --All together: %d--\n", alltogetherPrimes);



    fclose(outfp);
    fclose(infp);
    return 0;
}


int isPrime(int num) {
    if (num <= 1) {
        return 0;
    }

    for (int i = 2; i <= num / 2; i++) {
        if (num % i == 0) {
            return 0;
        }
    }

    return 1;
}
