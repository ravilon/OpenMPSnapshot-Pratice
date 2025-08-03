#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <omp.h>
#include "dictionary-util.c"
#include "../globals.h"

int compare_candidates(FILE **file, char *password_hash, int verbose);
int run_chunk(char *password_hash, char **candidate_array, int chunk_size, int verbose);

/**
* dictionary_crack() - OpenMP Implementation
*
* The OpenMP implementation of the dictionary attack.
*
* @param password_hash is the hashed password string to crack.
* @param dictionary_path is the full path, including the dictionary filename.
* @param verbose is a flag for verbose mode. Set to 1 to enable.
* @return the result as an integer value, FOUND (0) or NOT_FOUND (1).
*/
int dictionary_crack(char *password_hash, char *dictionary_path, int verbose)
{
// Print input parameters 
if( verbose )
{
printf("\n>>> Using dictionary path: %s\n", dictionary_path);
print_password_hash(password_hash);
}

// Open file
FILE *file = fopen(dictionary_path, "r");

// Do calculation
int result = compare_candidates(&file, password_hash, verbose);

if(result == NOT_FOUND)
print_not_found(verbose);

// Cleanup
fclose(file);
return result;
}

/**
* compare_candidates() - comparing password_hash against hashed dictionary entires
* (OpenMP Implementation)
* 
* 1. Manages iterating through the dictionary file and initiating the has comparisons.
* 2. Returns the result value (FOUND or NOT_FOUND) and the plain text password, if found.
*
* @param file is a pointer to the dictionary file in memory.
* @param password_hash is the hashed password string to crack.
* @param verbose is a flag for verbose mode. Set to 1 to enable.
* @return the result as an integer value, FOUND (0) or NOT_FOUND (1).
*/
int compare_candidates(FILE **file, char *password_hash, int verbose)
{
char *line = NULL;
size_t len = 0;
ssize_t read;
int result = NOT_FOUND;
int counter = 0;
char * candidate_array[CHUNK_SIZE];

while ((read = getline(&line, &len, *file) != -1) && result == NOT_FOUND)
{
remove_new_line(line, &candidate_array[counter]);
if (counter++ == CHUNK_SIZE)
{
result = run_chunk(password_hash, candidate_array, CHUNK_SIZE, verbose);
counter=0;
}
}
// finish off remaining work that didn't fit in a chunk.
if (counter > 0)
{
result = run_chunk(password_hash, candidate_array, counter, verbose);
}
return result;
}

/**
* run_chunk() - does hash comaprison on a chunk of passwords read from disk
* (OpenMP Implementation)
* 
* 1. Manages iterating through the dictionary file and initiating the has comparisons.
* 2. Returns the result value (FOUND or NOT_FOUND) and the plain text password, if found.
*
* @param file is a pointer to the dictionary file in memory.
* @param password_hash is the hashed password string to crack.
* @param candidate_array is the array of candidate passwords.
* @param array_size is the size of the candidate array.
* @param verbose is a flag for verbose mode. Set to 1 to enable.
* @return the result as an integer value, FOUND (0) or NOT_FOUND (1).
*/
int run_chunk(char *password_hash, char **candidate_array, int array_size, int verbose)
{
int result = NOT_FOUND;
int i;
#pragma omp parallel
#pragma omp for schedule(auto)
for (i=0; i<array_size; i++) 
{
int tempResult = do_comparison(password_hash, candidate_array[i], verbose);
if (tempResult == FOUND) {
#pragma omp critical
result = FOUND;
} 
} 
return result;
}
