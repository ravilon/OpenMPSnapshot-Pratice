#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif
#define FILENAME "input2.txt"
#define MAX_FILE_NAME 100
typedef struct
{
    int id;
    double x;
    double y;
} Ppoint;
void validateBcast(int res, int line, char sr[5]);
int main(int argc, char **argv);
int readInputFile(double **numbers, int **ids, int *numNPoints, int *numKPoints, double *distance, int *tCount);
void broadcasting(double **numbers, int **newIds, int *numNPoints, int *numKPoints, double *distance, int *tCount, int rank);
Ppoint *settingUpPoints(double *inputNums, int *ids, int n, int k, double distance, int world_rank, int t);
Ppoint *find3KNearestPoints(Ppoint *points, int n, double k, double distance, int countOfPoints);
int foundK(Ppoint *points, Ppoint checkedP, int k, int n, double distance, int count3Found);
void sendPoints(int *foundPointsIds, int actualTs);
void write_to_file(const char *filename, int tCount, int *ids);
int *recievePoints(int *myValues, int world_size, int tCount);

int main(int argc, char **argv)
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    double t1, t2;
    t1 = MPI_Wtime();
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);
    int n, k, tCount;
    double *inputNums;
    int *ids;
    double distance;
    if (world_rank == 0)
    {

        readInputFile(&inputNums, &ids, &n, &k, &distance, &tCount);
    }
    // #pragma omp parallel for
    //     for (int i = 0; i < 4 * (n); i++)
    //     {
    //         printf("numbers[%d] = %lf\n", i, (inputNums)[i]);
    //     }
    broadcasting(&inputNums, &ids, &n, &k, &distance, &tCount, world_rank);

    Ppoint *points;
    int tid, t;
    int numofThreads;
    int count3Found = 0;

    if (world_rank == world_size - 1)
        numofThreads = (tCount / world_size + tCount % world_size) + 1;
    else
        numofThreads = (tCount / world_size);

    int countOfPoints = 0;
    int *foundPointsFinalIds;

    foundPointsFinalIds = (int *)malloc(3 * numofThreads * sizeof(int));
    if (!foundPointsFinalIds)
    {
        printf("\nCannot allocate memory for foundPointsFinal array\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        exit(EXIT_FAILURE);
    }

#pragma omp parallel private(tid, t, points) shared(inputNums, ids, n, k, distance, count3Found, countOfPoints, foundPointsFinalIds) num_threads(numofThreads)
    {
        Ppoint foundPoints[3];
        tid = omp_get_thread_num();
        t = tid + world_rank * numofThreads;

        points = settingUpPoints(inputNums, ids, n, k, distance, world_rank, t);
#pragma omp barrier
printf("findnearestpoints t=%d\n", t);
        Ppoint *check = find3KNearestPoints(points, n, k, distance, countOfPoints);
        printf("check ended for t= %d\n", t);
        if (check != NULL)
        {
            memcpy(foundPoints, check, sizeof(foundPoints));
            for (int i = 0; i < 3; i++)
            {
                printf("for t = %d, %d \n", t, foundPoints[i].id);
            }
            free(check);
        }
        else
        {
#pragma omp for schedule(dynamic)
            for (int i = 0; i < 3; i++)
            {
                foundPoints[i].id = -1;
            }
        }
        free(points);
        printf("findnearestpoints FINISH t=%d\n", t);
#pragma omp barrier
        for (int i = 0; i < 3; i++)
        {
            foundPointsFinalIds[i + 3 * tid] = foundPoints[i].id;
            printf("for t = %d tid+i in array %d,value %d, process %d\n", t, i + 3 * tid, foundPointsFinalIds[i + 3 * tid], world_rank);
        }
    }
    for (int i = 0; i < numofThreads * 3; i++)
    {
        printf("for t = %d, %d for process %d\n", numofThreads * world_rank + i, foundPointsFinalIds[i], world_rank);
    }
    printf("process %d finished threads\n", world_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0)
    {

        int *overall = recievePoints(foundPointsFinalIds, world_size, tCount);
        write_to_file("output.txt", tCount, overall);
        printf("success writing output\n");
        free(overall);
    }
    else
    {
        sendPoints(foundPointsFinalIds, numofThreads);
        free(foundPointsFinalIds);
        printf("slave %d finished\n", world_rank);
    }

    t2 = MPI_Wtime();
    printf("Elapsed time is %f\n", t2 - t1);
    free(inputNums);
    free(ids);
    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
int readInputFile(double **numbers, int **ids, int *numNPoints, int *numKPoints, double *distance, int *tCount)
{
    FILE *fp = fopen(FILENAME, "r");
    if (!fp)
    {
        printf("Could not open the file\n");
        return 0;
    }
    printf("---------------------\n");
    printf("Starting to Read Input File\n");

    // read first line
    if (fscanf(fp, "%d %d %lf %d", numNPoints, numKPoints, distance, tCount) != 4)
    {
        printf("Failed to read number of pictures\n");
        fclose(fp);
        return 0;
    }
    *numbers = (double *)malloc((*numNPoints) * 4 * sizeof(double));
    if (*numbers == NULL)
    {
        printf("Memory allocation failed.\n");
        fclose(fp);
        return 0;
    }
    double *numbersOriginal = *numbers;
    *ids = (int *)malloc((*numNPoints) * sizeof(int));
    if (*ids == NULL)
    {
        printf("Memory allocation failed.\n");
        fclose(fp);
        return 0;
    }
    int *idsOriginal = *ids;
    for (size_t i = 0, j = 0; i < *numNPoints && j < 4 * (*numNPoints); i++, j += 4)
    {
        if (fscanf(fp, "%d %lf %lf %lf %lf", &(*ids)[i], &((*numbers)[j]), &((*numbers)[j + 1]), &((*numbers)[j + 2]), &((*numbers)[j + 3])) != 5)
        {
            printf("Failed to read numbers\n");
            fclose(fp);
            return 0;
        }
    }

    printf("N %d \nK %d\nD %lf\n tcount %d", *numNPoints, *numKPoints, *distance, *tCount);
    // set to first value
    *ids = idsOriginal;
    *numbers = numbersOriginal;
    fclose(fp);
    printf("\nFinished Reading Input File\n");
    return 1;
}

void broadcasting(double **numbers, int **ids, int *numNPoints, int *numKPoints, double *distance, int *tCount, int rank)
{
    char send[5] = "send", recv[5] = "recv";
    if (rank == 0)
    {
        validateBcast(MPI_Bcast(numNPoints, 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, send);
        validateBcast(MPI_Bcast(numKPoints, 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, send);
        validateBcast(MPI_Bcast(distance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, send);
        validateBcast(MPI_Bcast(tCount, 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, send);

        validateBcast(MPI_Bcast((*ids), *numNPoints, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, send);
        validateBcast(MPI_Bcast((*numbers), (*numNPoints) * 4, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, send);
    }
    else
    {
        validateBcast(MPI_Bcast(numNPoints, 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, recv);
        validateBcast(MPI_Bcast(numKPoints, 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, recv);
        validateBcast(MPI_Bcast(distance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, recv);
        validateBcast(MPI_Bcast(tCount, 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, recv);

        int *newIds = (int *)malloc((*numNPoints) * sizeof(int));
        if (!newIds)
        {
            printf("\nCannot allocate memory for slave %d newIds array\n", rank);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
            exit(EXIT_FAILURE);
        }

        double *newNumbers = (double *)malloc((4 * (*numNPoints)) * sizeof(double));
        if (!numbers)
        {
            printf("\nCannot allocate memory for slave %d numbers array\n", rank);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
            exit(EXIT_FAILURE);
        }

        validateBcast(MPI_Bcast((newIds), *numNPoints, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, recv);
        validateBcast(MPI_Bcast((newNumbers), (*numNPoints) * 4, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, recv);
        *ids = newIds;
        *numbers = newNumbers;
        printf("Finished broadcasting\n");
    }
}

Ppoint *settingUpPoints(double *inputNums, int *ids, int n, int k, double distance, int world_rank, int t)
{
    Ppoint *points = (Ppoint *)malloc(n * sizeof(Ppoint));
    if (!points)
    {
        printf("\nCannot allocate memory for %d points array\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        exit(EXIT_FAILURE);
    }
    // #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        // Print inputNums and ids for debugging purposes

        // printf("inputNums[%d] = %lf\n", i, inputNums[i]);
        // printf("ids[%d] = %d\n", i, ids[i]);

        points[i].id = *(ids + i);
        double x1 = *(inputNums + i * 4);
        double x2 = *(inputNums + i * 4 + 1);
        double a = *(inputNums + i * 4 + 2);
        double b = *(inputNums + i * 4 + 3);
        // printf("Point %d x1 %lf x2 %lf a %lf b %lf\n", points[i].id, x1, x2, a, b);
        points[i].x = ((x2 - x1) / 2) * sin(t * M_PI / 2) + ((x2 + x1) / 2);
        points[i].y = a * points[i].x + b;
        printf("Point %d x %lf y %lf\n", points[i].id, points[i].x, points[i].y);
    }

    return points;
}

Ppoint *find3KNearestPoints(Ppoint *points, int n, double k, double distance, int countOfPoints)
{
    int count3Found = 0;
    Ppoint *foundPoints = (Ppoint *)malloc(3 * sizeof(Ppoint));
    if (!foundPoints)
    {
        printf("\nCannot allocate memory for foundPoints array\n");
        return NULL;
    }

#pragma omp parallel for schedule(dynamic) default(shared) reduction(+ : count3Found)
    for (int j = 0; j < n; j++)
    {
        Ppoint checkedP = points[j];
        if (foundK(points, checkedP, k, n, distance, count3Found) < k)
            continue;
#pragma omp critical
        {
            if (count3Found < 3)
            {
                foundPoints[count3Found] = checkedP;
                count3Found++;
            }
        }
    }
    if (count3Found < 3)
    {
        free(foundPoints);
        return NULL;
    }
    return foundPoints;
}

void validateBcast(int res, int line, char sr[5])
{
    switch (res)
    {
    case MPI_SUCCESS:
        printf("%s bcast data successfully %d\n", sr, line);
        break;
    case MPI_ERR_COMM:
        printf("%s MPI_ERR_COMM %d\n", sr, line);
        break;
    case MPI_ERR_COUNT:
        printf("%s MPI_ERR_COUNT %d\n", sr, line);
        break;
    case MPI_ERR_TYPE:
        printf("%s MPI_ERR_TYPE %d\n", sr, line);
        break;
    case MPI_ERR_BUFFER:
        printf("%s MPI_ERR_BUFFER %d\n", sr, line);
        break;
    case MPI_ERR_ROOT:
        printf("%s MPI_ERR_ROOT %d\n", sr, line);
        break;
    default:
        printf("%s other... %d\n", sr, line);
        break;
    };
}

int foundK(Ppoint *points, Ppoint checkedP, int k, int n, double distance, int count3Found)
{
    int kCount = 0;

#pragma omp parallel for schedule(dynamic) default(shared) reduction(+ : kCount)
    for (int i = 0; i < n; i++)
    {
        if (points[i].id != checkedP.id)
        {
            double dist = sqrt(pow(points[i].x - checkedP.x, 2) + pow(points[i].y - checkedP.y, 2));
            if (dist < distance)
            {
                kCount++;
            }
        }
    }
    printf("kCount %d\n", kCount);
    return kCount;
}
void sendPoints(int *foundPointsIds, int actualTs)
{
    MPI_Gatherv(foundPointsIds, actualTs * 3, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
}
void write_to_file(const char *filename, int tCount, int *ids)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error: Failed to open file %s\n", filename);
        return;
    }

    for (int i = 0; i <= tCount; i++)
    {
        printf("\n\nwriting i %d\n\n", i);

        if (ids[i * 3] < -1)
        {
            printf("invalid id in index " + i * 3, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
            exit(EXIT_FAILURE);
        }
        else if (ids[i * 3] == -1)
        {
            continue;
        }
        else
        {
            fprintf(fp, "Points  pointID%d, pointID%d, pointID%d satisfy Proximity Criteria at t = %d \n", ids[i * 3], ids[i * 3 + 1], ids[i * 3 + 2], i);
        }
    }
    if (ftell(fp) == 0)
    {
        fprintf(fp, "There were no 3 points found for any t");
    }
    fclose(fp);
}
int *recievePoints(int *myValues, int world_size, int tCount)
{

    int counts[world_size];
    int displacements[world_size];
    for (int i = 0; i < world_size; i++)
    {
        // Define the receive counts
        // Define the displacements
        if (i == world_size - 1)
        {
            counts[i] = 3 * ((tCount / world_size) + (tCount % world_size) + 1);
            displacements[i] = i * 3 * (tCount / world_size);
        }
        else
        {
            counts[i] = 3 * (tCount / world_size);
            displacements[i] = i * 3 * (tCount / world_size);
        }
    }
    int *buffer = (int *)calloc((1 + tCount) * 3, sizeof(int));
    MPI_Gatherv(myValues, tCount / world_size * 3, MPI_INT, buffer, counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Values gathered in the buffer on process 0");
    for (int i = 0; i < tCount * 3; i++)
    {
        printf(" %d", buffer[i]);
    }
    printf("\n");
    return buffer;
}