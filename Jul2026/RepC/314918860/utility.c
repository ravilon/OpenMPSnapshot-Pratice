#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "omp.h"

#include "utility.h"


/**
 * This function swaps two float pointers.
 * 
 * @param xp: The first float pointer
 * @param yp: The second float pointer
 ****/ 
void swap(float* xp, float* yp) {
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}

/**
 * This functions copies a float array
 * 
 * @param bloc: The float array to be copied
 * @param bloc_size: The size of the float array
 * 
 * @return The copie of the input float array
 ****/ 
float* copy_bloc(float** bloc, int bloc_size) {
    float* new_b = (float*) malloc(sizeof(float) * bloc_size);
    for (size_t i = 0; i < bloc_size; i++)
        *(new_b+i) = *(*bloc+i);
    return new_b;
}

/**
 * This function sorts an array of 'k' floats 
 * using the selection sort algorithm.
 * 
 * @param bloc: The float array to be sorted
 * @param k: The size of the array
 *****/ 
void selection_sort(float **bloc, int k) {
    // Sort the copied bloc
    for (size_t i = 0; i < k; i++) {
        int min_idx = i;
        for (int j = i + 1; j < k; j++) {
            if (*(*bloc+j) < *(*bloc+min_idx))
                min_idx = j;
        }
        swap(*bloc+min_idx, *bloc+i);
    }
}

/**
 * This function heapifies a subtree rooted with 
 * node i which is an index in **bloc. 
 * 
 * @param bloc: The array to be sorted
 * @param bloc_size: The size of the array
 * @param i: The index 
 ***/ 
void heapify(float **bloc, int bloc_size, int i) {
    int largest = i; // init largest as root
    int l = 2 * i + 1; // left
    int r = 2 * i + 2; // right
    
    // If left child is larger than root
    if (l < bloc_size && **bloc+l > **bloc+largest)
        largest = l;
    
    // If right child is larger than largest so far
    if (r < bloc_size && **bloc+r > **bloc+largest)
        largest = r;
    
    // If largest is not root
    if (largest != i) {
        swap(*bloc+i, *bloc+largest);

        // Recursively heapify the affected sub-tree
        heapify(bloc, bloc_size, largest);
    }
}

/**
 * This function sorts an array using the `heap sort` algorithm.
 * 
 * @param bloc: The float array to be sorted
 * @param bloc_size: The size of the array
 ***/ 
void heap_sort(float** bloc, int bloc_size) {
    // Build heap (rearrange the array)
    for (int i = bloc_size / 2 - 1; i >= 0; i--)
        heapify(bloc, bloc_size, i);
    
    // Extract an element from heap (one by one)
    for (int i = bloc_size - 1; i > 0; i--) {
        // Move current root to end
        swap(*bloc, *bloc+i);

        // Call max heapify on the reduced heap
        heapify(bloc, i, 0);
    }
}

/**
 * This function sorts a bloc of floats.
 * 
 * @param bloc: The float array to be sorted
 * @param k: The size of the array
 *****/
void tri(float **bloc, int k){
    heap_sort(bloc, k);
}

/**
 * This function sorts two blocs in a way that the elements of 
 * the first bloc will be inferior the those in the second bloc.
 * 
 * @param bloc1: First bloc
 * @param bloc2: Second bloc
 * @param k: The size of the blocs
 *****/
void tri_merge(float **bloc1, float **bloc2, int k) {
    float* new_bloc = (float*) malloc(sizeof(float) * k*2);

    // Merge the two input blocs into a single bloc
    for (size_t i = 0; i < k*2; i++) {
        if (i < k) *(new_bloc+i) = *(*bloc1+i);
        else *(new_bloc+i) = *(*bloc2+i-k);
    }

    // Sort the new bloc (merged from bloc1 & bloc2)
    tri(&new_bloc, k*2);

    // Seperate the new bloc to two blocs (small numbers & big numbers)
    for (size_t i = 0; i < k; i++){
        *(*bloc1+i) = *(new_bloc+i);
        *(*bloc2+i) = *(new_bloc+i+k);
    }
    free(new_bloc);
}

/**
 * This function generates and returns a table of K floats randomly.
 * 
 * @param k: The size of the generated numbers
 **/
float* generator(int k) {
    float* bloc = (float*) malloc(sizeof(float) * k);

    for (size_t i = 0; i < k; i++)
        *(bloc+i) = (float)rand()/RAND_MAX;

    return bloc;
}

/**
 * This function frees the database out of the memory.
 * 
 * @param db: The database to free (two dimensional array)
 * @param n: The number of rows of the database
 **/
void free_db(float** db, int n) {
    for (size_t i = 0; i < n; i++)
        free(*(db+i));
    free(db);
}

/**
 * This function sorts a database in a parallel way.
 * ~ Following the given pseudocode from the teacher.
 * 
 * @param db: The database to be sorted (two dimensional array)
 * @param n: The number of rows of the database
 * @param k: The size of each row of the database
 ***/ 
void parallel_sort(float*** db, int n, int k, performance_measures* pm) {
    double t1 = omp_get_wtime();
    #pragma omp parallel for firstprivate(k)
    for (size_t i = 0; i < n; i++) 
        tri(*db+i, k);
    double t2 = omp_get_wtime();
    pm->first_sort = t2-t1;

    pm->second_sort = 0;
    for (size_t j = 0; j < n; j++) {
        int bi = 1 + (j % 2);
        t1 = omp_get_wtime();
        #pragma omp parallel for firstprivate(bi, n, k)
        for (size_t i = 0; i < n/2; i++) {
            int b1 = (bi + 2 * i) % n;
            int b2 = (bi + 2 * i+1) % n;
            int min = MIN(b1, b2);
            int max = MAX(b1, b2);
            tri_merge(*db+min, *db+max, k);
        }
        t2 = omp_get_wtime();
        pm->second_sort += t2-t1;
    }
}

/**
 * This function generates the statistics of the algorithms in a csv file.
 * 
 * @param filename: The output filename
 * @param nk: The matrix NxK (database for multiple tests) (two dimensional array)
 * @param nk_size: The size of the database (how many input)
 * @param nb_threads: The threads table [1, 2, 4, 8]
 * @param nb_threads_size: The size of the threads table 
 ***/ 
void generate_performances(char* filename, int nk[][2], int nk_size, int nb_threads[], int nb_threads_size) {
    printf("Running tests...\n");
    FILE* fp = NULL;
    fp = fopen(filename, "w");
    fprintf(fp, "N,K,NxK,#Threads,FirstSort,SecondSort,Performance\n");
    for (size_t i = 0; i < nk_size; i++) {
        int n = nk[i][0];
        int k = nk[i][1];
        printf("\n%5s | %5s | %8s | %7s | %13s | %13s | %13s\n",
            "N", "K", "NxK", "Threads", "1st Sort", "2nd Sort", "Total");
        for (size_t j = 0; j < nb_threads_size; j++) {
            int nb_th = nb_threads[j];
            printf("%5d | %5d | %8d | %7d | %13s | %13s | %13s", 
                n, k, n*k, nb_th, "CALCULATING..", "CALCULATING..", "CALCULATING..");
            fflush(stdout);
            performance_measures pm = get_performance_measures(n, k, nb_th);
            printf("\33[2K\r%5d | %5d | %8d | %7d | %13f | %13f | %13f\n", 
                n, k, n*k, nb_th, pm.first_sort, pm.second_sort, pm.sorting_span);
            fprintf(fp, "%d,%d,%d,%d,%f,%f,%f\n", 
                n, k, n*k, nb_th, pm.first_sort, pm.second_sort, pm.sorting_span);
            fflush(fp); // to write/save each line in the file (in case of interruption)
        }
        printf("\n");
    }
    fclose(fp);
    printf("%s file has been created successfully!\n", filename);
}

/**
 * This function generates a database, run the parallel sorting algorithms and
 * stores the performance analysis of the parallel sections.
 * 
 * @param n: The number of rows of the database
 * @param k: The number of columns of the database
 * @param nb_threads: The number of threads that the parallel section will use.
 * 
 * @return peformance_measures
 *****/ 
performance_measures get_performance_measures(int n, int k, int nb_threads) {
    performance_measures pm;
    srand((unsigned) time(NULL));
    
    // Set the number of threads
    omp_set_num_threads(nb_threads);
    
    // Generate 
    double t1 = omp_get_wtime();
    float** db = (float**) malloc(sizeof(float*) * n);
    for (size_t i = 0; i < n; i++) {
        *(db+i) = NULL;
        *(db+i) = generator(k);        
    }
    
    // Creating a copy of the db
    float** sorted_db = (float**) malloc(sizeof(float*) * n);
    for (size_t i = 0; i < n; i++) {
        float* bloc = (float*) malloc(sizeof(float) * k);
        for (size_t j = 0; j < k; j++) 
            *(bloc+j) = db[i][j];
        *(sorted_db+i) = bloc;
    }
    double t2 = omp_get_wtime();
    pm.generating_span = t2 - t1;

    // Sorting in parallel
    t1 = omp_get_wtime();
    parallel_sort(&sorted_db, n, k, &pm);
    t2 = omp_get_wtime();

    pm.sorting_span = t2 - t1;
    pm.total_span = pm.generating_span + pm.sorting_span;

    #if DEBUG==1
    // print the difference between tables
    printf("UNSORTED DATABASE\n");
    d_dump_db(db, n, k);
    printf("-----------------------------------\n");
    printf("SORTED DATABASE\n");
    d_dump_db(sorted_db, n, k);
    printf("\n\n");
    #endif

    free_db(sorted_db, n);
    free_db(db, n);
    return pm;
}

/**
 * This function prints out the database (two dimensional array)
 * 
 * @param db: The database (two dimensional array)
 * @param n: The number of rows
 * @param k: The number of columns
 ***/ 
void d_dump_db(float** db, int n, int k) {
    for (size_t i = 0; i < n; i++) {
        printf("db[%ld]\n", i);
        for (size_t j = 0; j < k; j++) {
            printf("%.2f|", *(*(db+i)+j));
        }
        printf("\n\n");
    }
}

/**
 * This function checks if a string (str1) contains another string (str2)
 * 
 * @param haystack: The string to search in
 * @param needle: The string that is to be searched
 * 
 * @return -1 if NOT_FOUND otherwise it's the position of str2 in str1
 ***/ 
int strpos(char *haystack, char *needle) {
    char *p = strstr(haystack, needle);
    if (p)
        return p - haystack;
    return -1;
}