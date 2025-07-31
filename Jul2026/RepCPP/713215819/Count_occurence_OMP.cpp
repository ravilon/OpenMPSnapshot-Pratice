#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <omp.h>
#include <bits/stdc++.h>
#include <chrono>	

#define MAX_WORD_LENGTH 100

using namespace std;

typedef struct WordCount {
    char word[MAX_WORD_LENGTH];
    int count;
} WordCount;

int compareWordCounts(const void *a, const void *b) {
    return ((WordCount *)b)->count - ((WordCount *)a)->count;
}

int main() {
	cout<<"Dhivyesh R K 2021BCS0084"<<endl;
    const char *filename = "test_file.txt";
    FILE *file = fopen(filename, "r");

    if (!file) {
        fprintf(stderr, "Error: Could not open the file %s\n", filename);
        return 1;
    }

    int max_word_count = 1000;
    WordCount *word_counts = (WordCount *)malloc(max_word_count * sizeof(WordCount));
    int num_words = 0;

    if (!word_counts) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return 1;
    }

    char word[MAX_WORD_LENGTH];
	auto  startSerial = chrono::high_resolution_clock::now();
    #pragma omp parallel private(word)
    {
        while (fscanf(file, "%99s", word) == 1) {
            
            int word_length = strlen(word);
            if (word_length == 0) {
                continue;
            }

            bool found = false;

            #pragma omp critical
            {
                for (int i = 0; i < num_words; i++) {
                    if (strcmp(word, word_counts[i].word) == 0) {
                        word_counts[i].count++;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    if (num_words >= max_word_count) {
                        max_word_count *= 2;
                        word_counts = (WordCount *)realloc(word_counts, max_word_count * sizeof(WordCount));
                        if (!word_counts) {
                            fprintf(stderr, "Error: Memory reallocation failed\n");
                            fclose(file);
                            exit(1);
                        }
                    }
                    strcpy(word_counts[num_words].word, word);
                    word_counts[num_words].count = 1;
                    num_words++;
                }
            }
        }
    }

    fclose(file);

    qsort(word_counts, num_words, sizeof(WordCount), compareWordCounts);
    for (int i = 0; i < num_words; i++) {
        printf("%s: %d\n", word_counts[i].word, word_counts[i].count);
    }
    free(word_counts);
	auto endSerial = chrono::high_resolution_clock::now();
	chrono::duration<double> durationSerial = endSerial - startSerial;
    double executionTimeSerial = durationSerial.count();
	cout<<"Time Taken : "<<executionTimeSerial<<endl;
  
    return 0;
}
