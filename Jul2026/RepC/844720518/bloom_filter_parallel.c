/**
 * @file    bloom_filter_parallel.c
 * @brief   Parallel implementation of Bloom Filter.
 * @author  Aflah Hanif Amarlyadi (aflahamarlyadi@gmail.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "bloom_filter.h"
#include "sha256.h"
#include "murmurhash3.h"

int isDistinct(char *word, char **array, int arrayLength) {
    for (int i = 0; i < arrayLength; i++) {
        if (strcmp(word, array[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

int readInsertFile(const char *filename, char ***wordArray, int *numWords) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    char word[MAX_WORD_LENGTH];
    int wordCount = 0;

    while (fscanf(file, "%s", word) != EOF) {
        wordCount++;
    }

    rewind(file);

    char **tempWordArray = (char**)malloc(wordCount * sizeof(char*));

    for (int i = 0; i < wordCount; i++) {
        fscanf(file, "%s", word);
        tempWordArray[i] = strdup(word);
    }

    fclose(file);

    *wordArray = tempWordArray;
    *numWords = wordCount;

    return wordCount;
}

int readQueryFile(const char *filename, char ***wordArray, int **labelArray) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }
    
    char word[MAX_WORD_LENGTH];
    int label;
    int wordCount = 0;

    while (fscanf(file, "%s %d", word, &label) != EOF) {
        wordCount++;
    }

    rewind(file);

    char **tempWordArray = (char **)malloc(wordCount * sizeof(char *));
    int *tempLabelArray = (int *)malloc(wordCount * sizeof(int));

    for(int i = 0; i < wordCount; i++){
        fscanf(file, "%s %d", word, &label);
        tempWordArray[i] = strdup(word);
        tempLabelArray[i] = label;
    }

    fclose(file);

    *wordArray = tempWordArray;
    *labelArray = tempLabelArray;

    return wordCount;
}

void insert(char *str, unsigned char *bitArray, unsigned int m, const int numHashes, HashType hashType) {
    if (hashType == HASH_SHA256) {
        for (int i = 0; i < numHashes; i++) {
            uint8_t hash[32];
            sha256(str, hash);
            uint32_t *hash_part = (uint32_t*)(hash + i * sizeof(uint32_t));
            uint32_t index = (*hash_part) % m;
            bitArray[index] = 1;
        }
    } else if (hashType == HASH_MURMURHASH3) {
        for (int i = 0; i < numHashes; i++) {
            uint32_t hash = murmurhash3(str, strlen(str), i);
            uint32_t index = hash % m;
            bitArray[index] = 1;
        }
    }
}

int query(char *str, unsigned char *bitArray, unsigned int m, const int numHashes, HashType hashType) {
    if (hashType == HASH_SHA256) {
        for (int i = 0; i < numHashes; i++) {
            uint8_t hash[32];
            sha256(str, hash);
            uint32_t *hash_part = (uint32_t*)(hash + i * sizeof(uint32_t));
            uint32_t index = (*hash_part) % m;

            if (bitArray[index] == 0) {
                return 0;
            }
        }
    } else if (hashType == HASH_MURMURHASH3) {
        for (int i = 0; i < numHashes; i++) {
            uint32_t hash = murmurhash3(str, strlen(str), i);
            uint32_t index = hash % m;

            if (bitArray[index] == 0) {
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <hash_function> <insert_file1> ... <insert_fileN>\n", argv[0]);
        return 1;
    }

    HashType hashType;
    if (strcmp(argv[1], "sha256") == 0) {
        hashType = HASH_SHA256;
        printf("Bloom Filter with SHA256 - Parallel\n");
    } else if (strcmp(argv[1], "murmurhash3") == 0) {
        hashType = HASH_MURMURHASH3;
        printf("Bloom Filter with MurmurHash3 - Parallel\n");
    } else {
        fprintf(stderr, "Invalid hash function. Use 'sha256' or 'murmurhash3'.\n");
        return 1;
    }

    int numInsertFiles = argc - 2;
    const char **insertFilenames = (const char **)malloc(numInsertFiles * sizeof(char *));
    for (int i = 0; i < numInsertFiles; i++) {
        insertFilenames[i] = argv[i + 2];
    }

    const char *queryFilename = "../data/query.txt";
    int numQueryWords;

    struct timespec start, end;
    double time_taken, total_time = 0;


    printf("\nPerformance:\n");


    // Read the insert files into arrays
    clock_gettime(CLOCK_MONOTONIC, &start);

    char **insertWords[numInsertFiles];
    int numInsertWords[numInsertFiles];
    int totalInsertWords = 0;

    // Each thread reads an insert file into an array in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numInsertFiles; i++) {
        totalInsertWords += readInsertFile(insertFilenames[i], &insertWords[i], &numInsertWords[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    total_time += time_taken;
    printf("Time taken for reading the insert files into arrays:    %lf s\n", time_taken);


    // Count the number of distinct words
    clock_gettime(CLOCK_MONOTONIC, &start);

    char **distinctWords = (char **)malloc(totalInsertWords * sizeof(char *));
    int numDistinctWords = 0;

    for (int i = 0; i < numInsertFiles; i++) {
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < numInsertWords[i]; j++) {
            if (!isDistinct(insertWords[i][j], distinctWords, numDistinctWords)) {
                #pragma omp critical
                {
                    distinctWords[numDistinctWords] = insertWords[i][j];
                    numDistinctWords++;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    total_time += time_taken;
    printf("Time taken for counting the number of distinct words:   %lf s\n", time_taken);


    // Calculate the optimal size of the bit array
    unsigned int optimalSize = (unsigned int)(-(numDistinctWords * log(MAX_FP_RATE)) / (pow(log(2), 2)));

    // Initialise the bit array
    unsigned char *bitArray = (unsigned char *)calloc(optimalSize, sizeof(unsigned char));

    // Calculate the optimal number of hashes
    const int numHashes = (int)round((optimalSize / (double)numDistinctWords) * log(2));


    // Insert words into the bit array
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Each thread inserts words into the bit array in parallel
    for (int i = 0; i < numInsertFiles; i++) {
        #pragma omp parallel for
        for (int j = 0; j < numInsertWords[i]; j++) {
            insert(insertWords[i][j], bitArray, optimalSize, numHashes, hashType);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    total_time += time_taken;
    printf("Time taken for inserting words into the bit array:      %lf s\n", time_taken);


    // Reading the query file into arrays
    clock_gettime(CLOCK_MONOTONIC, &start);

    char **queryWords;
    int *queryLabels;
    numQueryWords = readQueryFile(queryFilename, &queryWords, &queryLabels);

    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    total_time += time_taken;
    printf("Time taken for reading the query file into an array:    %lf s\n", time_taken);


    // Query words from the bit array
    clock_gettime(CLOCK_MONOTONIC, &start);

    int falsePositives = 0;

    // Each thread queries words from the bit array and counts the number of false positives in parallel
    #pragma omp parallel for reduction(+:falsePositives)
    for (int i = 0; i < numQueryWords; i++) {
        int queryResult = query(queryWords[i], bitArray, optimalSize, numHashes, hashType);
        if (queryResult == 1 && queryLabels[i] == 0) {
            falsePositives++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    total_time += time_taken;
    printf("Time taken for querying words from the bit array:       %lf s\n", time_taken);
    printf("Total time taken:                                       %lf s\n", total_time);
    

    printf("\nStatistics:\n");
    printf("Optimal size of the bit array:                          %u\n", optimalSize);
    printf("Optimal number of hashes:                               %u\n", numHashes);
    printf("Number of words inserted:                               %d\n", totalInsertWords);
    printf("Number of distinct words inserted (estimate):           %d\n", numDistinctWords);
    printf("Number of words queried:                                %u\n", numQueryWords);
    printf("Number of false positives:                              %d\n", falsePositives);
    printf("False positive rate:                                    %f%%\n", (double)falsePositives / numQueryWords * 100);


    // Clean up memory
    free(bitArray);

    free(distinctWords);

    for (int i = 0; i < numInsertFiles; i++) {
        for (int j = 0; j < numInsertWords[i]; j++) {
            free(insertWords[i][j]);
        }
        free(insertWords[i]);
    }
    
    for (int i = 0; i < numQueryWords; i++) {
        free(queryWords[i]);
    }
    free(queryWords);

    
    free(queryLabels);
    free(insertFilenames);

    return 0;
}
