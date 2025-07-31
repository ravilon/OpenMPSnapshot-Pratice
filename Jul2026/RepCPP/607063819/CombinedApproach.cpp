// Pristup koji kombinuje metode paralelnog izvrsavanja instrukcija pomocu OpenMP, SIMD paralelizma i dodatno je
// implemetiran mehanizam memoizacije radi ostvarenja jos boljih performansi pogotovo ako se program izvrsava vise puta.

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif                          // Da se ukinu neka upozorenja

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>
#include <immintrin.h>      // AVX2 instruction set


bool isPrime(int num);

char memo[1000000] = {0,0,1,1};                   // Ako je sizeof(char) = 1B onda mi ovo zauzima oko 1MB sto mi ne smeta 


// gcc -O3 -fopenmp -msse2 -o program program.c
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

    if (infp == NULL) return(printf("Error opening input file!\n"), -1);
    if (outfp == NULL) return(printf("Error opening output file!\n"), -1);


    fread(&numOfRanges, sizeof(int), 1, infp);

    for (int i = 0; i < numOfRanges; i++)
    {
        int count = 0;
        fread(&first, sizeof(int), 1, infp);
        fread(&last, sizeof(int), 1, infp);

        // Brojanje prostih brojeva
        #pragma omp parallel for reduction(+:count)
        for (int i = first; i <= last; i++) {
            if (memo[i]) {count++;}
            else if (isPrime(i)) { count++; memo[i] = 1; }           // Provjerava da li je prost samo ako vec ne znamo da li je prost
        }                                                            // i smjesta tu informaciju u memo
        alltogetherPrimes += count;

        // Upis u izlazni fajl
        fprintf(outfp, "The number of prime numbers between %d and %d is: %d\n", first, last, count);
    }

    fprintf(outfp, "\n     --All together: %d--\n", alltogetherPrimes);



    fclose(outfp);
    fclose(infp);
    return 0;
}


bool isPrime(int num)
{
    if (num <= 1 || num == 4 || num == 6)
        return false;

    if (num == 2 || num == 3 || num == 5)
        return true;


    __m128 number = _mm_setr_ps((float)num, (float)num, (float)num, (float)num);


    for (int i = 2; i < num / 2; i += 4)
    {
        __m128 iterationPair = _mm_setr_ps(i + 1, i + 2, i + 3, i + 4);
        __m128 result = _mm_div_ps(number, iterationPair);

        __m128i integerResult = _mm_cvttps_epi32(result);

        __m128 floatResult = _mm_cvtepi32_ps(integerResult);

        result = _mm_sub_ps(result, floatResult);
        result = _mm_cmpeq_ps(result, _mm_setzero_ps());

        int mask = _mm_movemask_ps(result);

        if (mask != 0) return false;
    }

    return true;
}

