//
//  main.c
//
//  Created by Jack Yang on 4/5/25.
//

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


void generatePrimes(int m, int n, int t, int *arr) {
    
    #pragma omp parallel for
    for (int i = m; i <= n; i++ ) {
        
        if (i < 2) continue;
        
        int prime = 1;
        
        for (int j = 2; j <= (i + 1) / 2; j++) {
            
            if (i % j == 0) {
                prime = 0;
                break;
            }
        }
        
        // Prime check passed
    
        if (prime == 1) {
            arr[i - m] = 1; // m is offset

        }
    }
}


int output(int *arr, int length, int offset, int name) {
    
    int primes = 0;
    
    char filename[64];
    snprintf(filename, sizeof(filename), "%d.txt", name);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Failed to open file");
        return 0;
    }

    for (int i = 0; i < length; i++) {
        if (arr[i] == 1) {
            fprintf(fp, "%d\n", i + offset);
            primes++;
        }
    }

    fclose(fp);
    return primes;
    
}





int main(int argc, const char * argv[]) {
    
    double tstart = 0.0, ttaken = 0.0;

    
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int t = atoi(argv[3]);
    
    
    omp_set_num_threads(t);
    
    int *arr = (int *)calloc( n - m + 1, sizeof(int));

    tstart = omp_get_wtime();
    generatePrimes(m, n, t, arr);
    ttaken = omp_get_wtime() - tstart;
    
    
    int primes = output(arr, n - m + 1, m, n);
    
    printf("The number of prime numbers found between %d and %d is %d\n", m, n, primes);
    printf("Time taken for the main part: %f\n", ttaken);
    
    free(arr);
    
    return 0;
}



