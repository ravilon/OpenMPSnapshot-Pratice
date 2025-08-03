#pragma once
using namespace std; //std::cout
#include <iostream> //cout, endl
#include <stdlib.h> //atoi, rand
#include <stdio.h> //printf
#include <omp.h> //OpenMP



void printDescriptorList(float* m, int matrixWidth, int matrixHeight) {

//printf("\n");
for (unsigned i = 0; i < matrixHeight; i++) {
for (unsigned j = 0; j < matrixWidth; j++) {
//printf("%f ", m[j + i * matrixWidth]);
}
//printf("\n");
}
//printf("\n");
}

void initDescriptorFloatsTwodigits(float* v, int vec_size) {

for (int i = 0; i < vec_size; i++) {
//v[i] = rand() % 100;
v[i] = (float)(rand() % 100) / 10;
//printf(" %f ", v[i]);
}
//printf("\n\n");
}

void printDescriptor(float* v, int n_elems) {

for (int i = 0; i < n_elems; i++) {
//printf(" % f ", v[i]);
}
//printf("\n\n");
}




//euclidean dist between two points
float euclDistBetweenTwoPoints(float point1, float point2) {

int ax = floor(point1);
int ay = (int)(point1 * 10) % 10;

int bx = floor(point2);
int by = (int)(point2 * 10) % 10;

float euclideanDist = (float)sqrt( (ax - bx)*(ax - bx) + (ay - by)*(ay - by) );

//printf("euclideanDist between %f and %f: %f\n", point1, point2, euclideanDist);

return euclideanDist;
}


//descriptor is a vector and descriptor_list is a matrix.
void getEuclideanDistancesVector(float* descriptor, float* descriptor_list, float* distances, int n_elems_descriptor, int n_descriptors) {

#pragma omp parallel for
for (int i = 0; i < n_descriptors; i++)
{
float sum = 0.0;
#pragma omp parallel for reduction (+:sum)
for (int j = 0; j < n_elems_descriptor; j++)
{
//printf("Thread %d calculating and adding euclidean dist between %f and %f\n", omp_get_thread_num(), descriptor_list[j + i * n_elems_descriptor], descriptor[j]);
float euclideanDist = euclDistBetweenTwoPoints(descriptor_list[j + i * n_elems_descriptor], descriptor[j]);
sum += euclideanDist;
//printf("Thread %d : descriptor %d: euclidean dist between %f and %f is: %f. Total Sum:%f \n", omp_get_thread_num(), i, descriptor_list[j + i * n_elems_descriptor], descriptor[j], euclideanDist, sum);
}
distances[i] = sum;
}
}


//Tamao de descriptor o vector de caractersticas.Evaluar diferentes tamaos.Se
//proponen vectores de 128, 256, 512.
//
//Tamao del ndice para la bsqueda.Evaluar el rendimiento empleando diferentes
//tamaos de ndice.Asumiremos ndices de 1024, 4096, 16384 y 65536 documentos.
void docSearchOpenMP(int n_elements_descriptor, int n_descriptors, int trials, int n_threads) {
printf("docSearch OpenMP");

int N_ELEMENTS_DESCRIPTOR = n_elements_descriptor;
int N_BYTES_DESCRIPTOR = N_ELEMENTS_DESCRIPTOR * sizeof(float);

int N_DESCRIPTORS = n_descriptors;
int N_BYTES_LIST_OF_DESCRIPTORS = N_ELEMENTS_DESCRIPTOR * N_DESCRIPTORS * sizeof(float);


//We reserve dynamic memory with malloc()
//malloc() returns a void* pointer so we have to cast it to the desired pointer type
float* descriptorToCompare = (float*)malloc(N_BYTES_DESCRIPTOR);
float* listOfDescriptors = (float*)malloc(N_BYTES_LIST_OF_DESCRIPTORS);
float* vectorOfDistances = (float*)malloc(N_DESCRIPTORS * sizeof(float)); //distances between each descriptor of the index and the descriptor to compare

initDescriptorFloatsTwodigits(descriptorToCompare, N_ELEMENTS_DESCRIPTOR); //vector
initDescriptorFloatsTwodigits(listOfDescriptors, N_ELEMENTS_DESCRIPTOR*N_DESCRIPTORS); //matrix

//Document search with OpenMP
omp_set_num_threads(n_threads);
double t1, t2;
t1 = omp_get_wtime();
for (int i = 0; i < trials; i++)
{
getEuclideanDistancesVector(descriptorToCompare, listOfDescriptors, vectorOfDistances, N_ELEMENTS_DESCRIPTOR, N_DESCRIPTORS);
}
t2 = omp_get_wtime();
printf("Tiempo medio en %d iteraciones de docSearchOpenMP() con %d threads, con vector de %d elementos e indice de %d descriptores: % lf seconds.\n\n", trials, n_threads, N_ELEMENTS_DESCRIPTOR, N_DESCRIPTORS, (t2 - t1) / (float)trials);


printf("descriptor to compare: ");
printDescriptor(descriptorToCompare, N_ELEMENTS_DESCRIPTOR);

printf("descriptor list: ");
printDescriptorList(listOfDescriptors, N_ELEMENTS_DESCRIPTOR, N_DESCRIPTORS);

printf("vector of euclidean distances: ");
printDescriptor(vectorOfDistances, N_DESCRIPTORS);

free(descriptorToCompare);
free(listOfDescriptors);
free(vectorOfDistances);
}