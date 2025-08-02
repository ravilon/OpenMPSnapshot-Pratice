#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>


#define n 1000000
#define d 5 // non-zero elements per col
#define b 5000 // block size = bxb

bool hasDuplicate(unsigned int* array, int length, int current);
void createMatrix(unsigned int* row, int nz_per_col);
void swap(int *xp, int *yp);
void bubblesort(unsigned int* array);
void mergeSort(int *arr, int l, int r, int *mirror);
double elapsed_time(struct timespec start, struct timespec end);

void BMM(int world_rank, int* chunk_offset, int* A_block_IDs, int* A_locations, int* A_nz_ptr, int* B_block_IDs, int* B_locations, 
            int* B_nz_ptr, int Blocks_per_row, int* A_nz_blocks_ptr, int* B_nz_blocks_ptr, int* C_i, int* C_j, int* C_nz);
void BMM_filtered(int world_rank, int* chunk_offset, int* A_block_IDs, int* A_locations, int* A_nz_ptr, int* B_block_IDs, int* B_locations, int* B_nz_ptr, int Blocks_per_row, 
            int* A_nz_blocks_ptr, int* B_nz_blocks_ptr, unsigned int* F_ptr, unsigned int* F_row, int* C_i, int* C_j, int* C_nz);
void convert_to_CSC(unsigned int* C_ptr, unsigned int* C_row, int* C_i, int* C_j, int C_nz);
void block_A(unsigned int* A_ptr, unsigned int* A_row, int* A_block_IDs, int* A_locations, 
            int* A_nz_ptr, int* A_nz_blocks_per_col, int* A_nz_blocks_ptr, int* nz_count_ptr, int* ID_count_ptr);
void block_B(unsigned int* B_ptr, unsigned int* B_row, int* B_block_IDs, int* B_locations, 
            int* B_nz_ptr, int* B_nz_blocks_per_col, int* B_nz_blocks_ptr, int* nz_count_ptr, int* ID_count_ptr);


int main(){
    srand(1);

    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int processes_count;
    MPI_Comm_size(MPI_COMM_WORLD, &processes_count);

    // ROWS
    unsigned int* A_row = (unsigned int *)malloc(d*n*sizeof(unsigned int));
    if (A_row==NULL) exit(-1);
    unsigned int* B_row = (unsigned int *)malloc(d*n*sizeof(unsigned int));
    if (B_row==NULL) exit(-1);
    unsigned int* F_row = (unsigned int *)malloc(100*n*sizeof(unsigned int));
    if (F_row==NULL) exit(-1);

    if(world_rank==0){
        printf("Creating matrices for testing...\n");
        createMatrix(A_row, d);
        createMatrix(B_row, d);
        createMatrix(F_row, 100);
    }
    MPI_Bcast(A_row, d*n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_row, d*n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(F_row, 100*n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // PTR
    unsigned int* A_ptr = (unsigned int *)malloc((n+1)*sizeof(unsigned int));
    if (A_ptr==NULL) exit(-1);
    A_ptr[0] = 1;
    for (int i=1; i<n+1; i++)
        A_ptr[i] = A_ptr[i-1] + d;

    unsigned int* B_ptr = (unsigned int *)malloc((n+1)*sizeof(unsigned int));
    if (B_ptr==NULL) exit(-1);
    B_ptr[0] = 1;
    for (int i=1; i<n+1; i++)
        B_ptr[i] = B_ptr[i-1] + d;

    unsigned int* F_ptr = (unsigned int *)malloc((n+1)*sizeof(unsigned int));
    if (F_ptr==NULL) exit(-1);
    F_ptr[0] = 1;
    for (int i=1; i<n+1; i++)
        F_ptr[i] = F_ptr[i-1] + 100;

    int total_blocks = (n/b)*(n/b);

    int* A_block_IDs = (int *)malloc(total_blocks*sizeof(int));
    if (A_block_IDs==NULL) exit(-1);

    int length_A_row = d*n;
    int* A_locations = (int *)malloc(3*length_A_row*sizeof(int));
    if (A_locations==NULL) exit(-1);

    int* A_nz_ptr = (int *)malloc(total_blocks*sizeof(int));
    if (A_nz_ptr==NULL) exit(-1);

    int* A_nz_blocks_per_row = (int *)malloc((n/b)*sizeof(int));
    if (A_nz_blocks_per_row==NULL) exit(-1);

    int* A_nz_blocks_ptr = (int *)malloc((n/b)*sizeof(int));
    if (A_nz_blocks_ptr==NULL) exit(-1);

    int* B_block_IDs = (int *)malloc(total_blocks*sizeof(int));
    if (B_block_IDs==NULL) exit(-1);

    int length_B_row = d*n;
    int* B_locations = (int *)malloc(3*length_B_row*sizeof(int));
    if (B_locations==NULL) exit(-1);

    int* B_nz_ptr = (int *)malloc(total_blocks*sizeof(int));
    if (B_nz_ptr==NULL) exit(-1);

    int* B_nz_blocks_per_col = (int *)malloc((n/b)*sizeof(int));
    if (B_nz_blocks_per_col==NULL) exit(-1);

    int* B_nz_blocks_ptr = (int *)malloc((n/b)*sizeof(int));
    if (B_nz_blocks_ptr==NULL) exit(-1);

    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec start;
    if(world_rank==0){
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    int nz_count_ptr_A, ID_count_ptr_A;
    int nz_count_ptr_B, ID_count_ptr_B;

    if(world_rank==0){
        printf("Blocking started.\n");
        block_A(A_ptr, A_row, A_block_IDs, A_locations, A_nz_ptr, A_nz_blocks_per_row, A_nz_blocks_ptr, &nz_count_ptr_A, &ID_count_ptr_A);
    }
    if(world_rank==1 || processes_count == 1)
        block_B(B_ptr, B_row, B_block_IDs, B_locations, B_nz_ptr, B_nz_blocks_per_col, B_nz_blocks_ptr, &nz_count_ptr_B, &ID_count_ptr_B);

    MPI_Bcast(&nz_count_ptr_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ID_count_ptr_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(A_block_IDs, ID_count_ptr_A, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(A_locations, 3*nz_count_ptr_A, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(A_nz_blocks_ptr, n/b, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(A_nz_ptr, ID_count_ptr_A, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    MPI_Bcast(&nz_count_ptr_B, 1, MPI_INT, 1, MPI_COMM_WORLD);
    MPI_Bcast(&ID_count_ptr_B, 1, MPI_INT, 1, MPI_COMM_WORLD);
    MPI_Bcast(B_block_IDs, ID_count_ptr_B, MPI_UNSIGNED, 1, MPI_COMM_WORLD);
    MPI_Bcast(B_locations, 3*nz_count_ptr_B, MPI_UNSIGNED, 1, MPI_COMM_WORLD);
    MPI_Bcast(B_nz_blocks_ptr, n/b, MPI_UNSIGNED, 1, MPI_COMM_WORLD);
    MPI_Bcast(B_nz_ptr, ID_count_ptr_B, MPI_UNSIGNED, 1, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec end;
    double time;
    if(world_rank==0){
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = elapsed_time(start, end);
        printf("Elapsed time (BLOCKING): %lf\n", time);
    }

    free(B_ptr); free(B_row);
    free(A_ptr); free(A_row);

    int* C_i;
    int* C_j;

    int C_nz = 0;
    int Blocks_per_row = n / b;

    if(world_rank==0){
        clock_gettime(CLOCK_MONOTONIC, &start);

        C_i = (int *)malloc(d*d*n*sizeof(int));
        if(C_i==NULL) exit(-1);

        C_j = (int *)malloc(d*d*n*sizeof(int));
        if(C_j==NULL) exit(-1);
    }

    int *chunk_offset = (int *)calloc(processes_count+1, sizeof(int));
    if(chunk_offset==NULL) exit(-1);

    if(world_rank==0){
        int chunk = total_blocks / processes_count;
        int remaining_blocks = total_blocks - chunk * processes_count;

        int *chunk_size = (int *)malloc(processes_count*sizeof(int));
        if(chunk_size==NULL) exit(-1);

        for(int process_iter=0; process_iter<processes_count; process_iter++){
            chunk_size[process_iter] = chunk;
        }

        while(remaining_blocks != 0){
            for(int process_iter=0; process_iter<processes_count; process_iter++){
                chunk_size[process_iter] += 1;
                remaining_blocks --;
                if(remaining_blocks == 0){
                    break;
                }
            }
        }
        for(int process_iter=1; process_iter<processes_count+1; process_iter++){
            chunk_offset[process_iter] = chunk_offset[process_iter-1] + chunk_size[process_iter-1];
        }
        free(chunk_size);

        printf("BMM started.\n");
    }
    MPI_Bcast(chunk_offset, processes_count+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //BMM(world_rank, chunk_offset, A_block_IDs, A_locations, A_nz_ptr, B_block_IDs, B_locations, B_nz_ptr, 
    //        Blocks_per_row, A_nz_blocks_ptr, B_nz_blocks_ptr, C_i, C_j, &C_nz);
    
    BMM_filtered(world_rank, chunk_offset, A_block_IDs, A_locations, A_nz_ptr, B_block_IDs, B_locations, B_nz_ptr, 
            Blocks_per_row, A_nz_blocks_ptr, B_nz_blocks_ptr, F_ptr, F_row, C_i, C_j, &C_nz);

    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank==0){
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = elapsed_time(start, end);
        printf("Elapsed time (BMM): %lf\n", time);
    }

    free(A_block_IDs);
    free(B_block_IDs);
    free(A_locations);
    free(B_locations);
    free(A_nz_ptr);
    free(B_nz_ptr);
    free(A_nz_blocks_per_row);
    free(B_nz_blocks_per_col);
    free(A_nz_blocks_ptr);
    free(B_nz_blocks_ptr);

    if(world_rank==0){
        printf("Converting C to CSC format...\n");

        unsigned int* C_row = (unsigned int*)malloc(C_nz*sizeof(unsigned int));
        if(C_row==NULL) exit(-1);

        unsigned int* C_ptr = (unsigned int*)malloc((n+1)*sizeof(unsigned int));
        if(C_ptr==NULL) exit(-1);
    
        clock_gettime(CLOCK_MONOTONIC, &start);

        convert_to_CSC(C_ptr, C_row, C_i, C_j, C_nz);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time = elapsed_time(start, end);
        printf("Elapsed time (CSC): %lf\n", time);
        free(C_row);
        free(C_ptr);
        free(C_i);
        free(C_j);
    }

    free(F_row);
    free(F_ptr);
    MPI_Finalize();
    return 0;
}


void block_A(unsigned int* A_ptr, unsigned int* A_row, int* A_block_IDs, int* A_locations, 
    int* A_nz_ptr, int* A_nz_blocks_per_row, int* A_nz_blocks_ptr, int *nz_count_ptr, int *ID_count_ptr){

    A_nz_ptr[0] = 0;
    
    int nz_count = 0;
    int ID_count = 0;
    int nz_blocks_per_row = 0;
    int current_row = 0;

    for(int p=0; p<(n/b); p++){ // iterate rows of blocks
        for(int q=0; q<(n/b); q++){ //iterate blocks of each row
            bool block_is_nz = false;
            for(int j=q*b; j<(q*b)+b; j++){ //iterate each column inside block
                for(int k=(A_ptr[j]-1); k<(A_ptr[j+1]-1); k++){ //iterate col_non_zeros
                    int current_row = A_row[k];
                    if(current_row >= p*b && current_row < (p+1)*b){
                        A_locations[3*nz_count] = current_row;
                        A_locations[3*nz_count + 1] = j;
                        A_locations[3*nz_count + 2] = j % b;
                        nz_count++;
                        if(block_is_nz == false){
                            A_block_IDs[ID_count] = p*(n/b) + q;
                            ID_count++;
                            nz_blocks_per_row++;
                            block_is_nz = true;
                        }
                    }
                    else if (A_row[k] >= (p+1)*b)
                        break;
                }
            }
            A_nz_ptr[ID_count] = nz_count;
        }
        A_nz_blocks_per_row[p] = nz_blocks_per_row;
        nz_blocks_per_row = 0;
    }
    A_nz_blocks_ptr[0] = 0;
    for(int i=1; i<(n/b); i++){
        A_nz_blocks_ptr[i] = A_nz_blocks_ptr[i-1] + A_nz_blocks_per_row[i-1];
    }

    printf("NZ_A = %d\n", nz_count);
    A_block_IDs = (int *)realloc(A_block_IDs, ID_count*sizeof(int));
    A_nz_ptr = (int *)realloc(A_nz_ptr, ID_count*sizeof(int));
    A_locations = (int *)realloc(A_locations, 3*nz_count*sizeof(int));
    *ID_count_ptr = ID_count;
    *nz_count_ptr = nz_count;
}


void block_B(unsigned int* B_ptr, unsigned int* B_row, int* B_block_IDs, int* B_locations, 
    int* B_nz_ptr, int* B_nz_blocks_per_col, int* B_nz_blocks_ptr, int *nz_count_ptr, int *ID_count_ptr){

    B_nz_ptr[0] = 0;

    int nz_count = 0;
    int ID_count = 0;
    int nz_blocks_per_col = 0;
    int current_row = 0;

    for(int q=0; q<(n/b); q++){ // iterate columns of blocks
        for(int p=0; p<(n/b); p++){ //iterate blocks of each column
            bool block_is_nz = false;
            for(int j=q*b; j<(q*b)+b; j++){ //iterate each column inside block
                for(int k=(B_ptr[j]-1); k<(B_ptr[j+1]-1); k++){ //iterate col_non_zeros
                    current_row = B_row[k];
                    if(current_row >= p*b && current_row < (p+1)*b){
                        B_locations[3*nz_count] = current_row;
                        B_locations[3*nz_count + 1] = j;
                        B_locations[3*nz_count + 2] = current_row % b; 
                        nz_count++;
                        if(block_is_nz == false){
                            B_block_IDs[ID_count] = q*(n/b) + p;
                            ID_count++;
                            nz_blocks_per_col++;
                            block_is_nz = true;
                        }
                    }
                    else if (B_row[k] >= (p+1)*b)
                        break;
                }
            }
            B_nz_ptr[ID_count] = nz_count;
        }
        B_nz_blocks_per_col[q] = nz_blocks_per_col;
        nz_blocks_per_col = 0;
    }
    B_nz_blocks_ptr[0] = 0;
    for(int i=1; i<(n/b); i++){
        B_nz_blocks_ptr[i] = B_nz_blocks_ptr[i-1] + B_nz_blocks_per_col[i-1];
    }

    printf("NZ_B = %d\n", nz_count);
    B_block_IDs = (int *)realloc(B_block_IDs, ID_count*sizeof(int));
    B_nz_ptr = (int *)realloc(B_nz_ptr, ID_count*sizeof(int));
    B_locations = (int *)realloc(B_locations, 3*nz_count*sizeof(int));
    *ID_count_ptr = ID_count;
    *nz_count_ptr = nz_count;
}


void convert_to_CSC(unsigned int* C_ptr, unsigned int* C_row, int* C_i, int* C_j, int C_nz){
    
    mergeSort(C_j, 0, C_nz-1, C_i);
    C_ptr[0] = 1;
    int ptr = 0;
    for(int i=0; i<n; i++){

        C_ptr[i+1] = C_ptr[i];

        for(int j=ptr; j<C_nz; j++){

            if (C_j[j] != i)
                break;
            else{
                C_row[C_ptr[i+1] - 1] = C_i[j];
                C_ptr[i+1]++;
                ptr++;
            }
        }
    }
}


void BMM_filtered(int world_rank, int* chunk_offset, int* A_block_IDs, int* A_locations, int* A_nz_ptr, int* B_block_IDs, int* B_locations, int* B_nz_ptr, 
            int Blocks_per_row, int* A_nz_blocks_ptr, int* B_nz_blocks_ptr, unsigned int* F_ptr, unsigned int* F_row, int* C_i, int* C_j, int* C_nz){
    
    int total_blocks = (n/b)*(n/b);
    int blocks_per_row = n / b;
    int nz_counter = 0;
    
    unsigned int* C_i_process = (unsigned int *)malloc(b*b*(chunk_offset[world_rank+1]-chunk_offset[world_rank])*sizeof(unsigned int));
    if(C_i_process==NULL) exit(-1);
    unsigned int* C_j_process = (unsigned int *)malloc(b*b*(chunk_offset[world_rank+1]-chunk_offset[world_rank])*sizeof(unsigned int));
    if(C_j_process==NULL) exit(-1);

    #pragma omp parallel for
    for (int block = chunk_offset[world_rank]; block<chunk_offset[world_rank+1]; block++){
        int blocks_row = block / blocks_per_row;
        int blocks_col = block % blocks_per_row;
        int block_nz = 0;

        unsigned int* C_i_block = (unsigned int *)malloc(b*b*sizeof(unsigned int));
        if(C_i_block==NULL) exit(-1);
        unsigned int* C_j_block = (unsigned int *)malloc(b*b*sizeof(unsigned int));
        if(C_j_block==NULL) exit(-1);

        // for each NZ block of A in "blocks_row" row
        for(int i = A_nz_blocks_ptr[blocks_row]; i<A_nz_blocks_ptr[blocks_row+1]; i++){

            int current_A_block = A_block_IDs[i];

            // for each NZ block of b in "blocks_col" column
            for(int j = B_nz_blocks_ptr[blocks_col]; j<B_nz_blocks_ptr[blocks_col+1]; j++){

                int current_B_block = B_block_IDs[j];

                if ((current_A_block%blocks_per_row) != (current_B_block%blocks_per_row)){
                    if ((current_A_block%blocks_per_row) < (current_B_block%blocks_per_row))
                        break;
                    continue;
                }
   
                // for each NZ bit in NZ block of A
                for(int nz_A = A_nz_ptr[i]; nz_A<A_nz_ptr[i+1]; nz_A++){
                    
                    int offset_A = A_locations[3*nz_A + 2];

                    // for each NZ bit in NZ block of B
                    for(int nz_B = B_nz_ptr[j]; nz_B<B_nz_ptr[j+1]; nz_B++){
                        int offset_B = B_locations[3*nz_B + 2];
                        
                        if(offset_A == offset_B){
                            // check if bit exists in F
                            for(int k = F_ptr[B_locations[3*nz_B + 1]]-1; k<F_ptr[B_locations[3*nz_B + 1]+1]-1; k++){
                                if(A_locations[3*nz_A] < F_row[k])
                                    break;
                                else if(A_locations[3*nz_A] == F_row[k]){
                                    C_i_block[block_nz] = A_locations[3*nz_A]; // == i_A
                                    C_j_block[block_nz] = B_locations[3*nz_B + 1]; // == j_B
                                    block_nz+=1;
                                }
                            }
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            for(int k = nz_counter; k<(nz_counter+block_nz); k++){
                C_i_process[k] = C_i_block[k - nz_counter];
                C_j_process[k] = C_j_block[k - nz_counter];
            }
            nz_counter += block_nz;
        }
    }
    C_i_process = (unsigned int *)realloc(C_i_process, nz_counter*sizeof(unsigned int));
    C_j_process = (unsigned int *)realloc(C_j_process, nz_counter*sizeof(unsigned int));
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Request request;
    if(world_rank != 0){
        // tags: 0 -> C_i, 1 -> C_j, 2 -> nz_counter
        MPI_Isend(C_i_process, nz_counter, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Isend(C_j_process, nz_counter, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, &request);
        MPI_Isend(&nz_counter, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &request);
    }
    else {
        // add process' own NZs
        for(int i = *C_nz; i<(*C_nz+nz_counter); i++){
            C_i[i] = C_i_process[i - *C_nz];
            C_j[i] = C_j_process[i - *C_nz];
        }
        *C_nz += nz_counter;
        int processes_count;
        MPI_Comm_size(MPI_COMM_WORLD, &processes_count);

        // receive other processes' NZs
        unsigned int *buffer_i;
        unsigned int *buffer_j;
        for(int i = 1; i<processes_count; i++){

            MPI_Recv(&nz_counter, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            buffer_i = (unsigned int *)malloc(nz_counter*sizeof(unsigned int));
            MPI_Recv(buffer_i, nz_counter, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            buffer_j = (unsigned int *)malloc(nz_counter*sizeof(unsigned int));
            MPI_Recv(buffer_j, nz_counter, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int k = *C_nz; k<(*C_nz+nz_counter); k++){
                C_i[k] = buffer_i[k - *C_nz];
                C_j[k] = buffer_j[k - *C_nz];
            }
            *C_nz += nz_counter;
        }
        C_i = (int *)realloc(C_i, *C_nz*sizeof(int));
        C_j = (int *)realloc(C_j, *C_nz*sizeof(int));
        printf("NZ_C = %d\n", *C_nz);
    }
}


void BMM(int world_rank, int* chunk_offset, int* A_block_IDs, int* A_locations, int* A_nz_ptr, int* B_block_IDs, int* B_locations, 
            int* B_nz_ptr, int Blocks_per_row, int* A_nz_blocks_ptr, int* B_nz_blocks_ptr, int* C_i, int* C_j, int* C_nz){
    
    int total_blocks = (n/b)*(n/b);
    int blocks_per_row = n / b;
    int nz_counter = 0;

    unsigned int* C_i_process = (unsigned int *)malloc(b*b*(chunk_offset[world_rank+1]-chunk_offset[world_rank])*sizeof(unsigned int));
    if(C_i_process==NULL) exit(-1);
    unsigned int* C_j_process = (unsigned int *)malloc(b*b*(chunk_offset[world_rank+1]-chunk_offset[world_rank])*sizeof(unsigned int));
    if(C_j_process==NULL) exit(-1);

    #pragma omp parallel for
    for (int block = chunk_offset[world_rank]; block<chunk_offset[world_rank+1]; block++){
        int blocks_row = block / blocks_per_row;
        int blocks_col = block % blocks_per_row;
        int block_nz = 0;

        unsigned int* C_i_block = (unsigned int *)malloc(b*b*sizeof(unsigned int));
        if(C_i_block==NULL) exit(-1);
        unsigned int* C_j_block = (unsigned int *)malloc(b*b*sizeof(unsigned int));
        if(C_j_block==NULL) exit(-1);

        // for each NZ block of A in "blocks_row" row
        for(int i = A_nz_blocks_ptr[blocks_row]; i<A_nz_blocks_ptr[blocks_row+1]; i++){

            int current_A_block = A_block_IDs[i];

            // for each NZ block of B in "blocks_col" column
            for(int j = B_nz_blocks_ptr[blocks_col]; j<B_nz_blocks_ptr[blocks_col+1]; j++){

                int current_B_block = B_block_IDs[j];

                if ((current_A_block%blocks_per_row) != (current_B_block%blocks_per_row)){
                    if ((current_A_block%blocks_per_row) < (current_B_block%blocks_per_row))
                        break;
                    continue;
                }
                    
                // for each NZ bit in NZ block of A
                for(int nz_A = A_nz_ptr[i]; nz_A<A_nz_ptr[i+1]; nz_A++){
                    
                    int offset_A = A_locations[3*nz_A + 2];

                    // for each NZ bit in NZ block of B
                    for(int nz_B = B_nz_ptr[j]; nz_B<B_nz_ptr[j+1]; nz_B++){
                        
                        int offset_B = B_locations[3*nz_B + 2];
                        
                        if(offset_A == offset_B){
                            C_i_block[block_nz] = A_locations[3*nz_A]; // == i_A
                            C_j_block[block_nz] = B_locations[3*nz_B + 1]; // == j_B
                            block_nz+=1;
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            for(int i = nz_counter; i<(nz_counter + block_nz); i++){
                C_i_process[i] = C_i_block[i - nz_counter];
                C_j_process[i] = C_j_block[i - nz_counter];
            }
            nz_counter += block_nz;
        }
    }
    C_i_process = (unsigned int *)realloc(C_i_process, nz_counter*sizeof(unsigned int));
    C_j_process = (unsigned int *)realloc(C_j_process, nz_counter*sizeof(unsigned int));
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Request request;
    if(world_rank != 0){
        // tags: 0 -> C_i, 1 -> C_j, 2 -> nz_counter
        MPI_Isend(C_i_process, nz_counter, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Isend(C_j_process, nz_counter, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, &request);
        MPI_Isend(&nz_counter, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &request);
    }
    else {
        // add process' own NZs
        for(int i = *C_nz; i<(*C_nz+nz_counter); i++){
            C_i[i] = C_i_process[i - *C_nz];
            C_j[i] = C_j_process[i - *C_nz];
        }
        *C_nz += nz_counter;
        int processes_count;
        MPI_Comm_size(MPI_COMM_WORLD, &processes_count);

        // receive other processes' NZs
        unsigned int *buffer_i;
        unsigned int *buffer_j;
        for(int i = 1; i<processes_count; i++){

            MPI_Recv(&nz_counter, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            buffer_i = (unsigned int *)malloc(nz_counter*sizeof(unsigned int));
            MPI_Recv(buffer_i, nz_counter, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            buffer_j = (unsigned int *)malloc(nz_counter*sizeof(unsigned int));
            MPI_Recv(buffer_j, nz_counter, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int k = *C_nz; k<(*C_nz+nz_counter); k++){
                C_i[k] = buffer_i[k - *C_nz];
                C_j[k] = buffer_j[k - *C_nz];
            }
            *C_nz += nz_counter;
        }
        C_i = (int *)realloc(C_i, *C_nz*sizeof(int));
        C_j = (int *)realloc(C_j, *C_nz*sizeof(int));
        printf("NZ_C = %d\n", *C_nz);
    }
}


void createMatrix(unsigned int* row, int nz_per_col){
    unsigned int* buffer; 
    for (int i=0; i<n; i++){
        buffer = (unsigned int *)calloc(nz_per_col, sizeof(unsigned int));
        if (buffer==NULL) exit(-1);

        for (int j=0; j<nz_per_col; j++){
            do{
                buffer[j] = (unsigned int)(rand() % n);
            } while(hasDuplicate(buffer, nz_per_col, j));
        }
        // sort before adding, for easier search
        bubblesort(buffer);
        for (int j=0; j<nz_per_col; j++){
            row[i*nz_per_col + j] = buffer[j];
        }
    }
    free(buffer);
}


void bubblesort(unsigned int* array){
    int i, j;
    for (i = 0; i < d-1; i++){
        // Last i elements are already in place
        for (j = 0; j < d-i-1; j++){
            if (array[j] > array[j+1])
                swap(&array[j], &array[j+1]);
        }
    }
}


void swap(int *xp, int *yp){
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

// Searches for duplicate element in an array
bool hasDuplicate(unsigned int* array, int length, int current){  
    for(int i = 0; i < current; i++) {    
        if(array[i] == array[current])    
            return true;      
    }
    return false; 
}


void merge(int *arr, int l, int m, int r,int *mirror){
    int k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L,*R,*L_mirror,*R_mirror;
    L=(int *) malloc(n1 * sizeof(int));
    R =(int *) malloc(n2 * sizeof(int));
    L_mirror=(int *) malloc(n1 * sizeof(int));
    R_mirror =(int *) malloc(n2 * sizeof(int));


    for (int i = 0; i < n1; i++){
        L[i] = arr[l + i];
        L_mirror[i] = mirror[l+i];
    }

    for (int j = 0; j < n2; j++){
        R[j] = arr[m + 1 + j];
        R_mirror[j] = mirror[m+1+j];
    }

    int i = 0;
    int j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            mirror[k] = L_mirror[i];
            i++;
        }
        else {
            arr[k] = R[j];
            mirror[k] = R_mirror[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        mirror[k] = L_mirror[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        mirror[k] = R_mirror[j];
        j++;
        k++;
    }
}


void mergeSort(int *arr, int l, int r, int *mirror){
    if (l < r) {
        int m = l + (r - l) / 2;
        
        //#pragma omp task
        mergeSort(arr, l, m, mirror);
        
        mergeSort(arr, m + 1, r,mirror);

        //#pragma omp taskwait
        merge(arr, l, m, r,mirror);
    }
}


// fuction to calculate elapsed time between two timespecs (same as the one in the PDS deliverables)
double elapsed_time(struct timespec start,struct timespec end){
        struct timespec temp;
        if ((end.tv_nsec - start.tv_nsec) < 0){
            temp.tv_sec = end.tv_sec - start.tv_sec - 1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        }
        else {
            temp.tv_sec = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        return (double)temp.tv_sec +(double)((double)temp.tv_nsec/(double)1000000000);
}
