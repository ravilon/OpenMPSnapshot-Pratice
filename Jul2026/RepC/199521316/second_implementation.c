#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BUFF_SIZE 50
#define MAXSIZE 100
#define MAXLINE 50
#define N_EXEC 100 //number of parallel executed k_means algorithms
#define N_FOLD 10 //number of folds for cross validation
#define K_MAX 10 //max number of cluster
#define CHUNKSIZE 10
#define IMPL 2
char FILEPATH[] = "data/iris_high.txt";

int k = 2; // initial value of clusters
int threads, chunkn, chunkm;

struct data
{
int dim;
int atts;
float** data;
};

//structure that contains all SSE and centroids computed
struct history
{
float SSE;
float** centroids;
};

float calcSilhouette(float** dataset, int **clusters, float** centroids, int n, int m);
struct data loadDataset(char* fileName, char* dist);
void normalize(struct data* dataset);
void datasetSubSets(struct data dataset, int fold, struct data* trainingSet, struct data* testSet);
float mainAlgo(struct data training, struct data test, int flagFinal);
void kmeans (struct data structure, int numIte, float tol, struct history* recordStoria);
void copySubMatrix(float** centroids, float** dataset, int *ranNum, int m);
void randomIndexes(int *ranNum, int n);
void zeroClusters(int **clusters, int n);
void findClusters(float** dataset, int **clusters, float** centroids, int n, int m);
float calcSSE(float** dataset, int **clusters, float** centroids, int n, int m);
void freeArray(float **a, int n);
void copyMatrix(float **mat1, float **mat2, int row, int col);
void printData(struct data dataset);
void getRow(float **matrix, int row, float *array, int m);
float eucliDist(float *rec1, float *rec2, int m);
void findCentroids(float** centroids, int **clusters, float** dataset, int n, int m);
void printClusters(int **clusters, int n);
void printCentroids(float** centroids, int m);
void writeFile(float** data, int **clusters, int n, int m);
void freeArrayInt(int **a, int n);
int omp_thread_count();
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
int main (int argc, char *argv[])
{

threads = omp_thread_count();
strcpy(FILEPATH, argv[1]);
char file[30] = "../data/";
strcat(file,argv[1]);
strcat(file,".csv");
double begin = omp_get_wtime();
double end;
struct data dataset = loadDataset(file, "\t");
chunkn = dataset.dim / threads;
chunkm = (dataset.atts / threads) + 1;
printf("\nCHUNKN: %d - CHUNKM: %d\n", chunkn, chunkm);
normalize(&dataset);
printf("DIM: %d\n", dataset.dim);
struct data trainingSet;
struct data testSet;
int bestk=2;
float sumSil[ K_MAX-1 ], appSSE, appSIL, SIL[ K_MAX-1 ];
for(k = 2; k<=K_MAX; k++) {
printf("\nAnalizing for k = %d", k);
sumSil[k-2] = 0;
//N_FOLD-fold cross validation
for(int fold=0; fold<N_FOLD; fold++) {
trainingSet.data = (float**) calloc (dataset.dim - (dataset.dim / N_FOLD), sizeof(float*));
testSet.data = (float**) calloc (dataset.dim / N_FOLD, sizeof(float*));
datasetSubSets(dataset, fold, &trainingSet, &testSet);
sumSil[k-2] +=mainAlgo(trainingSet, testSet, 0);
if(fold<N_FOLD-1){
freeArray(trainingSet.data, trainingSet.dim);
freeArray(testSet.data, testSet.dim);
}
}
sumSil[k-2] = sumSil[k-2]/N_FOLD;
//AIC[k-2] = 2*k + log10(sumSil[k-2]/testSet.dim); //Akaike criterion result
printf("\nSilhouette: %f",sumSil[k-2]);
//printf("\nAIC: %f, altro: %f", AIC[k-2], testSet.dim*log10(sumSil[k-2]/testSet.dim));
if(k==2)
appSIL = sumSil[k-2];
if(sumSil[k-2] > appSIL) {
bestk = k;
appSIL = sumSil[k-2];
}
freeArray(trainingSet.data, trainingSet.dim);
freeArray(testSet.data, testSet.dim);
end = omp_get_wtime();
double time_spent = (end - begin);
printf("\nTime from start: %lf sec \n------------------------", time_spent);
}
printf("\n best k is: %d with Silhuette: %f", bestk, appSIL);
// Setting the number of clusters to the best one chosen before as a result of AIC compare
k = bestk;
mainAlgo(dataset, dataset, 1);
end = omp_get_wtime();

FILE* fd;
fd = fopen("tempi.txt", "a");
fprintf(fd,"\n%lf sec\t%d\t%s\t%d",(end - begin), omp_get_num_threads(), FILEPATH, IMPL);
printf("\nk_max= %d, Total time: %lf sec\n",K_MAX,(end - begin));
}

//main algorithm
//flag is =1 only when final iteration is computed (in order to write out the results)
float mainAlgo(struct data training, struct data test, int flagFinal) {
struct history bestStoria;
float supportSSE;
int** bestClusters =(int**) calloc(test.dim, sizeof(int*));
bestStoria.centroids = (float**) malloc(k * sizeof(float*));
for(int i=0; i<test.dim; i++)
bestClusters[i] =(int*) calloc(k, sizeof(int));
for(int i=0; i<k; i++)
bestStoria.centroids[i] =(float*) malloc(training.atts *sizeof(float));
bestStoria.SSE = training.atts * training.dim;
struct history storia;
storia.centroids = (float**) calloc(k, sizeof(float*));
for(int i=0; i<k; i++) {
storia.centroids[i] =(float*) calloc(training.atts, sizeof(float));
}
srand( time(NULL) );
for(int i=0; i<N_EXEC; i++) {
kmeans(training, 5000, 0.001, &storia);
if(storia.SSE <= bestStoria.SSE) {
copyMatrix(bestStoria.centroids, storia.centroids, k, training.atts);
bestStoria.SSE = storia.SSE;
}
}
zeroClusters(bestClusters, test.dim); //reset best clusters matrix
findClusters(test.data, bestClusters, bestStoria.centroids, test.dim, test.atts);
//print last iteration results
if(flagFinal == 1) {
printCentroids(bestStoria.centroids, test.atts);
writeFile(test.data,  bestClusters, test.dim, test.atts);
}

float SSEtrovato = calcSSE(test.data, bestClusters, bestStoria.centroids, test.dim, test.atts);
float silTrovata = calcSilhouette(test.data, bestClusters, bestStoria.centroids, test.dim, test.atts);
//printf("\nthread: %d", omp_get_thread_num());
freeArray(bestStoria.centroids, k);
freeArray(storia.centroids, k);
freeArrayInt(bestClusters, test.dim);
return silTrovata;
}

float calcSilhouette(float** dataset, int **clusters, float** centroids, int n, int m){
float sum=0.0, supDataset[m], supCentroid[m], avgi[k], avge[k], minAvge = 10, a, b, max = 0, sil, meansil = 0;
int trovato = 0;
int ci = 0;
for(int ki=0; ki<k; ki++){
getRow(centroids, ki, supCentroid, m);
avgi[ki] = 0;
max = 0;
for(int i=0;i<n;i++){
getRow(dataset, i, supDataset, m);
ci += clusters[i][ki];
avgi[ki] += eucliDist(supCentroid, supDataset, m) * clusters[i][ki];
if(trovato==0)
avge[ki] += eucliDist(supCentroid, supDataset, m) * (1 - clusters[i][ki]);
if(clusters[i][ki]==0) trovato = 1;
if(clusters[i][ki] == 0 && minAvge > eucliDist(supCentroid, supDataset, m)) minAvge = eucliDist(supCentroid, supDataset, m);
}
if(ci!=0)
avgi[ki] = avgi[ki] / ci;
trovato = 0;
ci=0;
}

float lowestAvge = minAvge, avgiMean=0;
for(int ki=0;ki<k;ki++){
if(lowestAvge>=avgi[ki]){
max = lowestAvge;
sil = 1-(avgi[ki]/lowestAvge);
}
else{
max = avgi[ki];
sil = (lowestAvge/avgi[ki])-1;
}
meansil += sil;
}
meansil = meansil / k;

return meansil;
}

void kmeans (struct data structure, int numIte, float tol, struct history* recordStoria)
{
int n, m, *ranNum;
n = structure.dim;
m = structure.atts;
ranNum = (int*) calloc(k, sizeof(int));
//generate k random indexes to start from
randomIndexes(ranNum, n);
float** centroids = (float**) calloc(k, sizeof(float*));
int** clusters =(int**) calloc(n, sizeof(int*));
for(int i=0; i<n; i++) {
clusters[i] =(int*) calloc(k, sizeof(int));
}
for(int i=0; i<k; i++) {
centroids[i] =(float*) calloc(m, sizeof(float));
}
//saving initial centroids
copySubMatrix(centroids, structure.data, ranNum, m);
/*saving indexes that corresponds to each of the k clusters.
Results will be stored into a matrix which columns are the
cluster number and the rows corresponds to the indexes of records */
zeroClusters(clusters, n); //pongo a 0 tutti gli elementi del cluster
findClusters(structure.data, clusters, centroids, n, m);
int count = 0;
float newSSE, currSSE;
float** supCentroids = (float**) calloc(k, sizeof(float*));
for(int j = 0; j < k; j++)
supCentroids[j] = (float*) calloc(m, sizeof(float));
do {
currSSE = calcSSE(structure.data, clusters, centroids, n, m);
copyMatrix(supCentroids, centroids, k, m);
findCentroids(centroids, clusters, structure.data, n, m);
zeroClusters(clusters, n);
findClusters(structure.data,clusters,centroids, n, m);
newSSE = calcSSE(structure.data, clusters, centroids, n, m); //sum of square errors calculation
count++;
} while(count < numIte && ((currSSE-newSSE)/currSSE) > tol);
if(newSSE > currSSE) {
copyMatrix(centroids, supCentroids, k, m);
newSSE = currSSE;
}
copyMatrix(recordStoria->centroids, centroids, k, m);
recordStoria->SSE = newSSE;
freeArray(centroids,k);
freeArrayInt(clusters,n);
freeArray(supCentroids,k);
free(ranNum);
}

void printData(struct data dataset) {
printf("\n");
for(int i = 0; i < dataset.dim; i++) {
printf("%d\t", i + 1);
for(int j = 0; j < dataset.atts; j++) {
printf("%.2f\t", dataset.data[i][j]);
}
printf("\n");
}
}

void normalize(struct data* dataset) {
int i, j;
printf("Normalizing the data\n");
float max[dataset->atts];
// Look for max of each column
for(i = 0; i < dataset->dim; i++) {
for(j = 0; j < dataset->atts; j++) {
if(i == 0) {
max[j] = 0;
}
if(max[j] < dataset->data[i][j])
max[j] = dataset->data[i][j];
}
}
// Normalize the data by dividing each value by the max value of the column
for(i = 0; i < dataset->dim; i++) {
for(j = 0; j < dataset->atts; j++) {
dataset->data[i][j] = dataset->data[i][j] / max[j];
}
}
}

void datasetSubSets(struct data dataset, int fold, struct data* trainingSet, struct data* testSet) {
int init, end, apptr = 0, appte=0;
init = fold * (dataset.dim / N_FOLD);
end = ((fold + 1) * (dataset.dim / N_FOLD)) - 1;
trainingSet->dim = 0;
trainingSet->atts = dataset.atts;
testSet->dim = 0;
testSet->atts = dataset.atts;
for (int i = 0; i < dataset.dim; i++) {
if(i >= init && i <= end) {
testSet->data[appte] = (float*) calloc(dataset.atts, sizeof(float));
for(int u=0;u<dataset.atts;u++)
testSet->data[appte][u] = dataset.data[i][u];
appte++;
} else {
trainingSet->data[apptr] = (float*) calloc(dataset.atts, sizeof(float));
for(int u=0; u<dataset.atts; u++)
trainingSet->data[apptr][u] = dataset.data[i][u];
apptr++;
}
}
testSet->dim = appte;
trainingSet->dim = apptr;
}

struct data loadDataset(char* fileName, char* dist) {
FILE *file;
int max = MAXSIZE;
float* app;
float** data;
struct data dataset;
// Open file and check for I/O errors
file = fopen(fileName, "r");
if (file == NULL)  exit(-1);
else printf("Loading from file: %s\n", fileName);
char buffer[MAXLINE];
// Count the number of attributes
if (fgets(buffer, sizeof(buffer), file)) {
// Get the number of attributes
sscanf(buffer, "%d", &dataset.atts);
printf("atts: %d\n", dataset.atts);
}
// Read the actual data from the file
dataset.dim = 0;
// Read line by line until there are more in the file
dataset.data = (float**) calloc (MAXSIZE, sizeof(float*));
while(fgets(buffer, sizeof(buffer), file)) {
dataset.data[dataset.dim] = (float*) calloc(dataset.atts, sizeof(float));
// Take the first token in the current line of data
char *headapp = strtok(buffer, dist);
int i = 0;
do {
// Get the token and convert it to float
dataset.data[dataset.dim][i] = atof(headapp);
// Get the next token
headapp = strtok(NULL, dist);
i++;
} while(headapp != NULL);
dataset.dim++;
if(dataset.dim > max) {
max += MAXSIZE;
dataset.data = (float**) realloc(dataset.data, sizeof(float) * dataset.atts * max);
}
}
return dataset;
}

void freeArray(float **a, int n) {
for (int i = 0; i < n; ++i) {
free(a[i]);
}
free(a);
}

void freeArrayInt(int **a, int n) {
for (int i = 0; i < n; ++i) {
free(a[i]);
}
free(a);
}

void copyMatrix(float **mat1, float **mat2, int row, int col){ //mat1 dest, mat2 source
for (int i = 0; i < row; i++) {
for (int j = 0; j < col; j++) {
mat1[i][j] =  mat2[i][j];
}
}
}

//copy datasets records that corrensponds to centroids
void copySubMatrix(float **centroids, float** dataset, int *ranNum, int m) {
for(int i=0; i<k; i++)
{
for(int j=0; j<m; j++)
centroids[i][j] = dataset[ ranNum[i] ][j];
}
}

float calcSSE(float** dataset, int **clusters, float** centroids, int n, int m){
float sum=0.0, supDataset[m], supCentroid[m];
for(int ki=0; ki<k; ki++){
getRow(centroids, ki, supCentroid, m);
#pragma omp parallel for private(supDataset) reduction(+:sum) schedule(static, chunkn)
for(int i=0;i<n;i++){
getRow(dataset, i, supDataset, m);
sum += eucliDist(supCentroid, supDataset, m) * clusters[i][ki];
}
}
return sum;
}

void printClusters(int **clusters, int n){
printf("\t\t");
for(int ki=0;ki<k;ki++){
printf("k%d\t",ki);
}
for(int i=0;i<n;i++){
printf("\nrecord%d:\t",i);
for(int j=0;j<k;j++){
printf("%d\t",clusters[i][j]);
}
}
printf("\n");
}

void findCentroids(float** centroids, int **clusters, float** dataset, int n, int m) {
int elemCluster=0;
float record[m];
//reset array
for(int ki=0; ki<k; ki++) {
for(int p=0; p<m; p++) {
record[p] = 0;
}
for(int i=0; i<n; i++) {
if(clusters[i][ki]!=0) {
elemCluster++;
for(int j=0; j<m; j++)
record[j] += dataset[i][j];
}
}
#pragma omp parallel for schedule(static, chunkm) if(m >= omp_get_num_threads())
for(int p=0; p<m; p++) {
if(elemCluster!=0)
record[p] = record[p]/elemCluster;
else
record[p]=0;
centroids[ki][p] = record[p];
}
elemCluster = 0;
}
}

void printCentroids(float** centroids, int m) {
int p, i;
for(i = 0; i < k; i++) {
printf("\ncentroide cluster %d esimo: ",i);
for(p=0;p<m;p++){
printf("%.2f,",centroids[i][p]);
}
printf("\n");
}
}

void zeroClusters(int **clusters, int n){
for(int i=0;i<n;i++){
for(int j=0;j<k;j++){
clusters[i][j] = 0;
}
}
}

//it assigns each dataset record to the nearest centroid, which corrensponds to its cluster
void findClusters(float** dataset, int **clusters, float **centroids, int n, int m){
int salvaK=0;
float supCentroid[m], supDataset[m], dist=0, lowerDist;
#pragma omp parallel for schedule(static, chunkn) private(lowerDist, dist, salvaK, supCentroid, supDataset)
for(int i=0;i<n;i++){
lowerDist = m;
// lowerDist = m because data is normalized, so the max dist between 2 records is the number of attributes
for(int ki=0;ki<k;ki++){
getRow(centroids, ki, supCentroid, m); //extract a row from the centroid matrix
getRow(dataset, i, supDataset, m); //extract a row from the dataset matrix
dist = eucliDist(supCentroid, supDataset, m); //computing the euclidean distance
if(dist<=lowerDist)
{
lowerDist = dist;
salvaK = ki;
}
}
clusters[i][salvaK] = 1;
}
}

void getRow(float **matrix, int row, float *array, int m){
for(int j = 0; j < m; j++){
array[j] = matrix[row][j];
}
}

/*function that generate k indexes, which will refer to k initial centroids*/
void randomIndexes(int *ranNum, int n){
int i;
for(i=0;i<k;i++){
ranNum[i] = rand()%n;
}
}

/*receive two records in input so it calculates euclidean distance, continuos data required */
float eucliDist(float *rec1, float *rec2, int m){
float sum = 0.0;
for(int i=0;i<m;i++){
sum += ((rec1[i]-rec2[i]))*((rec1[i]-rec2[i]));
}
return sqrt(sum);
}

void writeFile(float** data ,int **clusters, int n, int m) {
FILE* fd;
char output[30] = "out/";
strcat(output,FILEPATH);
strcat(output,".txt");
fd = fopen(output, "w");
for(int i = 0; i < n; i++) {
fprintf(fd, "%d", i);
for(int j = 0; j < k; j++) {
fprintf(fd, "\t%d", clusters[i][j]);
}
fprintf(fd,"\n");
}
}

int omp_thread_count() {
int n = 0;
#pragma omp parallel reduction(+:n)
n += 1;
printf("\nNUM THREADS: %d\n", n);
return n;
}
