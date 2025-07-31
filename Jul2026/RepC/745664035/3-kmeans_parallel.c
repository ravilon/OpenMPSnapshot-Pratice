
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// NB Classes
#define NB_CLASSES 4

// NB points in the dataset
#define NB_POINTS 10000

// 3d point
typedef struct
{
    double x1;
    double x2;
    double x3;

} Point;

long long timespec_diff_ns(struct timespec *start, struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) * 1000000000LL + (end->tv_nsec - start->tv_nsec);
}

// generate dataset
void remplirPoints(Point *points, int num_points)
{
    int i;
    for (i = 0; i < num_points; i++)
    {
        points[i].x1 = rand() % 5;
        points[i].x2 = rand() % 5;
        points[i].x3 = rand() % 5;
    }
}

// initialisation of center of each classes
void initialiser(Point *centres, int K, int num_points, Point *points)
{
    int i, a, p = 2;
    for (i = 0; i < K; i++)
    {
        a = num_points / p;
        centres[i].x1 = points[a].x1;
        centres[i].x2 = points[a].x2;
        centres[i].x3 = points[a].x3;
        p++;
    }
}

// All points are not in a class
int InitialiserClasses(int *classe, int num_points)
{
    int i;
    for (i = 0; i < num_points; i++)
    {
        classe[i] = -1;
    }
}

// distance measurment
double calculerDistance(Point point1, Point point2)
{
    return sqrt((pow((point1.x1 - point2.x1), 2) + pow((point1.x2 - point2.x2), 2) + pow((point1.x3 - point2.x3), 2)));
}

// find best suited class
int pointClasse(Point point, Point *centres, int K)
{
    int parent = 0;
    double dist = 0;
    double minDist = calculerDistance(point, centres[0]);
    int i;
    for (i = 1; i < K; i++)
    {
        dist = calculerDistance(point, centres[i]);
        if (minDist >= dist)
        {
            parent = i;
            minDist = dist;
        }
    }
    return parent;
}

// calculate new center of each class
void calcNouvCentre(Point *points, int *classe, Point *centres, int K, int num_points)
{
    Point *nouvCentre = malloc(sizeof(Point) * K);
    int *members = malloc(sizeof(int) * K);
    int i;

    for (i = 0; i < K; i++)
    {
        members[i] = 0;
        nouvCentre[i].x1 = 0;
        nouvCentre[i].x2 = 0;
        nouvCentre[i].x3 = 0;
    }

    for (i = 0; i < num_points; i++)
    {
        members[classe[i]]++;
        nouvCentre[classe[i]].x1 += points[i].x1;
        nouvCentre[classe[i]].x2 += points[i].x2;
        nouvCentre[classe[i]].x3 += points[i].x3;
    }

    for (i = 0; i < K; i++)
    {
        if (members[i] != 0.0)
        {
            nouvCentre[i].x1 /= members[i];
            nouvCentre[i].x2 /= members[i];
            nouvCentre[i].x3 /= members[i];
        }
        else
        {
            nouvCentre[i].x1 = 0;
            nouvCentre[i].x2 = 0;
            nouvCentre[i].x3 = 0;
        }
    }
    for (i = 0; i < K; i++)
    {
        centres[i].x1 = nouvCentre[i].x1;
        centres[i].x2 = nouvCentre[i].x2;
        centres[i].x3 = nouvCentre[i].x3;
    }
}

// check if the array is stable
int stable(int *anciennes_classes, int *nouvelles_classes, int num_points, float tol)

{
    int i;
    int stable = 1; // variable to indicate if the array is stable
    tol = num_points * tol;

#pragma omp for
    for (i = 0; i < num_points; i++)
    {
        if (!stable)
            continue; // skip iterations if instability is already found

        if (abs(anciennes_classes[i] - nouvelles_classes[i]) > tol)
        {
#pragma omp critical
            stable = 0; // set stable to 0 if instability is found
        }
    }

    return stable ? 0 : -1;
}

int main(int argc, char *argv[])
{
    struct timespec start, end;
    long long elapsed_ns;
    FILE *fp = fopen("kmeans_results_4.csv", "w");
    fprintf(fp, "Points,Temps\n");

    for (int num_points = 10; num_points < 2000000; num_points += 5000)
    {
        int K; // Nombre de classes à construire
        // int num_points; // Nombre de points dans le dataset
        int i;
        int job_done = 0; // Fin
        float tol;        // La tolérance pour la stabilité
        float timeInS;
        clock_t t1, t2;

        Point *centres; // Tableau contenant les centres de classes
        Point *points;  // Dataset

        int *anciennes_classes;
        int *nouvelles_classes;
        int thread_num;

        srand(1);
        K = NB_CLASSES;
        // num_points = NB_POINTS;
        tol = 0;

        // Créer l'ensemble de points

        points = (Point *)malloc(sizeof(Point) * num_points);
        remplirPoints(points, num_points);

        // Alocation des classes
        anciennes_classes = (int *)malloc(sizeof(int) * num_points);
        nouvelles_classes = (int *)malloc(sizeof(int) * num_points);
        centres = malloc(sizeof(Point) * K);

        // Initialiser
        initialiser(centres, K, num_points, points);
        InitialiserClasses(anciennes_classes, num_points);
        InitialiserClasses(nouvelles_classes, num_points);

        clock_gettime(CLOCK_MONOTONIC, &start);

        omp_set_num_threads(4);
#pragma omp parallel
        {
            while (!job_done)
            {
#pragma omp for
                for (i = 0; i < num_points; i++)
                    nouvelles_classes[i] = pointClasse(points[i], centres, K);

#pragma omp single
                {

                    calcNouvCentre(points, nouvelles_classes, centres, K, num_points);
                }

                int stable_flag = 0;

                stable_flag = stable(nouvelles_classes, anciennes_classes, num_points, tol);

#pragma omp barrier

                if (stable_flag)
                {

                    job_done = 1;
                }
                else
                {
                    for (i = 0; i < num_points; i++)
                        anciennes_classes[i] = nouvelles_classes[i];
                }
            }

            clock_gettime(CLOCK_MONOTONIC, &end);

            elapsed_ns = timespec_diff_ns(&start, &end);

            // for (i = 0; i < num_points; i++)
            //     printf("Le point: %d,%d,%d appartient à la classe : %d\n", (int)points[i].x1, (int)points[i].x2, (int)points[i].x3, nouvelles_classes[i] + 1);

            // for (i = 0; i < K; i++)
            //     printf("Le centre de la classe %d : %d,%d,%d\n", i + 1, (int)centres[i].x1, (int)centres[i].x2, (int)centres[i].x3);
        }
        printf("%d\n", num_points);
        fprintf(fp, "%d,%ld\n", num_points, elapsed_ns);
    }
    fclose(fp);

    return 0;
}
// ******************************************************************************
// Code parallèle à reporter ci-dessous...

// Nom/Prénom_Etudiant 1 :
// Nom/Prénom_Etudiant 2 :
// ******************************************************************************
