//
//  matrix-projet.c
//  
//
//  Created by Aurélien Spinelli on 04/04/2018.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

struct matrix {
    int height;
    int width;
    int * tab;
};

struct matrix * allocateMatrix(int height, int width) {
    struct matrix * tmp = malloc(sizeof(struct matrix));
    tmp->height = height;
    tmp->width = width;
    tmp->tab = (int *)calloc(height * width, sizeof(int));
    return tmp;
}

int getValue(struct matrix * m, int i, int j){
    return m->tab[i * m->width + j];
}

void setValue(struct matrix * m, int i, int j, int value){
    m->tab[i * m->width + j] = value;
}

void printMatrix(struct matrix * m){
    for(int i = 0; i < m->height; i++){
        for(int y = 0; y < m->width; y++){
            printf("%d ", getValue(m, i, y));
        }
        printf("\n");
    }
}

// We read only the first line to know the size of matrix -> MATRIX N*N
struct matrix * generateMatrixFromFile(FILE * fp){
    // It is assumed that the file is well formed
    int x;
    int N = 0;
    while(fscanf(fp, "%d", &x) != EOF){
        N++;
        if(fgetc(fp) == 10){
            break;
        }
    }
    rewind(fp); // We put the buffer at the beginning of the file
    struct matrix * tmp;
    tmp = allocateMatrix(N, N);
    int numberOfInt = N * N;
    for(int i = 0; i < numberOfInt; i++){
        fscanf(fp, "%d", &tmp->tab[i]);
    }
    return tmp;
}

void multMatrix(struct matrix * a, struct matrix * b, struct matrix * c, int startingLineIndex){
    //  The matrix c will be the matrix calculate by our processor fills as one goes along
    //  It will have the same dimensions as B
    //  We make A x B: Order matters!
    int N = a->width; //or b->height;
    #pragma omp parallel for
    for(int i = 0; i < a->height; i++){
        for(int j = 0; j < b->width; j++){
            for(int k = 0; k < N; k++){
                int tmp = getValue(c, i + startingLineIndex, j) + getValue(a, i, k) * getValue(b, k, j);
                setValue(c, i + startingLineIndex, j, tmp);
            }
        }
    }
}

/*
 The scatter init can be done even if the number of lines is divisible by the number of processor, but be careful for the tab of B: We will use vectors.
 For example :
 - vector: we define a line and we send it several times numbersB = [2, 2, 2, 2, 3],
   displsB = [0, 2, 4, 6, 8]. This is the same principle as for A except that we do not reason on the integer number but on the number of vectors.
 */
void initForScatterv(int * countsA, int * displsA, int * countsB, int * displsB, int N, int num_procs){
    int mod = N % num_procs; // Number of processors with one more line
    int startingIndice = num_procs - mod;
    int numberOfLine = N / num_procs; // Number of lines per processor
    int dispA = 0;
    int dispB = 0;
    for(int i = 0; i < num_procs; i++){
        countsA[i] = numberOfLine;
        displsA[i] = dispA;
        countsB[i] = numberOfLine;
        displsB[i] = dispB;
        if(i >= startingIndice){
            countsA[i] += 1;
            countsB[i] += 1;
        }
        dispB += countsA[i];
        countsA[i] *= N;
        dispA += countsA[i];
    }
}



int main(int argc, char *argv[]) {

//----------------------    INIT    ----------------------
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int suivant = (rank + 1) % num_procs;
    int precedent = (rank - 1 + num_procs) % num_procs;
    int N;
    struct matrix * a = malloc(sizeof(struct matrix));
    struct matrix * b = malloc(sizeof(struct matrix));
    
    if(rank == 0){
        FILE* fp = NULL;
        fp = fopen(argv[1], "r");
        free(a);
        a = generateMatrixFromFile(fp);
        fclose(fp);
        
        fp = fopen(argv[2], "r");
        free(b);
        b = generateMatrixFromFile(fp);
        fclose(fp);
        
        N = a->height;
    }
    
//----------------------    FIN INIT    ----------------------
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); // We send to everyone the size of the matrix (N*N)
    
    /*
     These tables are defined in the initForScatterv, they respectively allow to know:
           - the number of integers that we will receive for the lines of A
           - the offset in the array where the matrix A is stored (in integer number)
           - the number of columns (defined by the vectors below)
           - the column number offset for the matrix B
     */
    int countsA[num_procs];
    int displsA[num_procs];
    int countsB[num_procs];
    int displsB[num_procs];
    
    initForScatterv(countsA, displsA, countsB, displsB, N, num_procs);
    /*
     NumberOfItem is the number of integer that we will receive for A, dividing it by N we can know the number of lines
     */
    int numberOfItem = countsA[rank];
    
//    Before the rotations (just after the scatterv) the number of rows and columns are distributed in the same way
    int * a_tab = malloc(numberOfItem * sizeof(int));
    int * b_tab = malloc(numberOfItem * sizeof(int));
    
//    Sending of A
    MPI_Scatterv(a->tab, countsA, displsA, MPI_INT, a_tab, numberOfItem, MPI_INT, 0, MPI_COMM_WORLD);
//    Storage of A
    struct matrix * sub_a = allocateMatrix(numberOfItem / N, N);
    free(sub_a->tab);
    sub_a->tab = a_tab;
    
//----------------------    SENDING OF B   ----------------------
    /*
     We define two types: the type that defines the vector for the columns on the matrix N * N and a type that defines the vector for local columns after they are sliced. These types are useful for sending columns of matrix B and for recovery columns of local results with the gather.
     */
    MPI_Datatype type, column_t;
    MPI_Type_vector(N, 1, N, MPI_INT, &type);
    MPI_Type_commit(&type);
    MPI_Type_create_resized(type, 0, sizeof(int), &column_t);
    MPI_Type_commit(&column_t);
    
    MPI_Datatype local_type, local_column_t;
    MPI_Type_vector(N, 1, countsB[rank], MPI_INT, &local_type);
    MPI_Type_commit(&local_type);
    MPI_Type_create_resized(local_type, 0, sizeof(int), &local_column_t);
    MPI_Type_commit(&local_column_t);
    
    MPI_Scatterv(b->tab, countsB, displsB, column_t, b_tab, countsB[rank], local_column_t, 0, MPI_COMM_WORLD);
//    Storage of B
    struct matrix * sub_b = allocateMatrix(N, countsB[rank]);
    free(sub_b->tab);
    sub_b->tab = b_tab;
    
//----------------------    END SENDING OF B    ----------------------
    
//----------------------    START MULT CALCULATION WITH ROTATION   ----------------------
   
    int currentIndice = displsB[rank]; // Line number that we will calculate
    struct matrix * c = allocateMatrix(N, countsB[rank]); // Creating the local res matrix
    multMatrix(sub_a, sub_b, c, currentIndice); // First mult before rotation
    //    Rotation of A on the ring
    for(int i = 0; i < num_procs - 1; ++i) {
        int nextNumberOfItem = countsA[(rank - (i + 1) + num_procs) % num_procs]; // We know the next integer number that we will receive during the rotation
        if(rank % 2 == 0){
            MPI_Send(sub_a->tab, numberOfItem, MPI_INT, suivant, 0, MPI_COMM_WORLD);
            if(numberOfItem != nextNumberOfItem){
                free(sub_a->tab);
                free(sub_a);
                sub_a = allocateMatrix(nextNumberOfItem / N, N); // We allocate the right number of lines
                //free(sub_a->tab); // to do ?
            }
            MPI_Recv(sub_a->tab, nextNumberOfItem, MPI_INT, precedent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            int * value_to_receive = malloc(nextNumberOfItem * sizeof(int));
            MPI_Recv(value_to_receive, nextNumberOfItem, MPI_INT, precedent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sub_a->tab, numberOfItem, MPI_INT, suivant, 0, MPI_COMM_WORLD);
            if(numberOfItem != nextNumberOfItem){
                free(sub_a);
                sub_a = allocateMatrix(nextNumberOfItem / N, N);
                //                free(sub_a->tab); to do ?
            }
            free(sub_a->tab);
            sub_a->tab = value_to_receive;
        }
        currentIndice = displsB[(rank - (i + 1) + num_procs) % num_procs];
        numberOfItem = nextNumberOfItem;
        multMatrix(sub_a, sub_b, c, currentIndice);
//        printMatrix(c); // Print to see the evolution of c
    }
    
    int * res = NULL;
    if(rank == 0){
        res = malloc(N * N * sizeof(int));
    }
//    We gather all the defined vectors in the same way that we scatter
    
    MPI_Gatherv(c->tab, countsB[rank], local_column_t, res, countsB, displsB, column_t, 0, MPI_COMM_WORLD);
    if(rank == 0){
        struct matrix * r = allocateMatrix(N, N);
        free(r->tab);
        r->tab = res;
        printMatrix(r);
    }
    
//----------------------    END MULT CALCULATION WITH ROTATION    ----------------------
    
//    MPI_Type_free(&column_t);
    free(sub_a->tab);
    free(sub_a);
    MPI_Finalize();
    return 0;
}


