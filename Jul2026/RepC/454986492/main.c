#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void initClock(struct timeval *begin) {
// Get initial time
gettimeofday(begin, 0);
}

void endClock(struct timeval *end) {
// Get end time
gettimeofday(end, 0);
}

void getElapsedTime(struct timeval *begin, struct timeval *end) {
// Calculate second diff
long seconds = end->tv_sec - begin->tv_sec;
// Calculate microseconds diff
long microseconds = end->tv_usec - begin->tv_usec;
// Calculate seconds and microseconds addition
double elapsed = seconds + microseconds*1e-6;

// Print the elapsed time
printf("[TIME] %.3f sec.\n", elapsed);
}

void algSecuencial(long unsigned int iterations) {
// Result variable
double respuesta = 0.0;
// Number of iterations
long unsigned int numeroIteraciones = iterations;

// Proccess of principal task
for(long indice = 0; indice <= numeroIteraciones; indice++){
// Conditional to toggle between addition and subtraction 
if(indice % 2 == 0){
respuesta += 4.0 / (2.0 * indice + 1.0);
}else{
respuesta -= 4.0 / (2.0 * indice + 1.0);
}
}

// Presentation of the result
printf("[SECUENCIAL] %d | %.8f\n", numeroIteraciones, respuesta);
}

void algParalelo(long unsigned int iterations) {
// Threan number varibale and id thread
int numeroHilos = 4, idHilo;
// Set the numbers of thread
omp_set_num_threads(numeroHilos);
// Result variable, and list of partials operations
double respuesta = 0.0, sumasParciales[numeroHilos];
// Number of iterations
long unsigned int numeroIteraciones = iterations;

// Use the parallel directive, and shared the partial list
#pragma omp parallel private(idHilo) shared(sumasParciales)
{
// Get the number id  of this thead
int idHilo = omp_get_thread_num();
// Initialize the partial value in 0 decimal
sumasParciales[idHilo] = 0.0;

// Proccess of principal task
for(long indice = idHilo; indice < numeroIteraciones; indice += numeroHilos) {
// Conditional to toggle between addition and subtraction
if(indice % 2 == 0) {
sumasParciales[idHilo] += 4.0 / (2.0 * indice + 1.0);
} else {
sumasParciales[idHilo] -= 4.0 / (2.0 * indice + 1.0);
}
}
}

// Finall for to calculate the addition of the partial list
for(int indice = 0; indice < numeroHilos; indice++) {
// Added partial sum to response.
respuesta += sumasParciales[indice];
}

// Presentation of the result
printf("[PARALELO  ] %d | %.8f\n", numeroIteraciones, respuesta);
}

int main() {
// Measuring time variables
struct timeval begin, end;

// Iteraccions variables
long unsigned int iterations;

// Get number of iteraccions
printf("Numero de iteracciones: ");
scanf("%lud", &iterations);
printf("\n");

for(int i = 0; i < 10; i++) {

// Init elapsed time
initClock(&begin);
// Exec secuencial function
algSecuencial(iterations);
// End elapsed time
endClock(&end);
// Get elapsed time for secuencial function execution
getElapsedTime(&begin, &end);

printf("\n"); // Spacer

// Init elapsed time
initClock(&begin);
// Exec parallel function
algParalelo(iterations);
// End elapsed time
endClock(&end);
// Get elapsed time for parallel function execution
getElapsedTime(&begin, &end);

long unsigned int increment;
if(i == 0 || i % 2 == 0) {
increment = iterations / 2;
}

// Increment a zero to the number of iterations 10 times
iterations = iterations + increment;

printf("\n\n"); // Spacer
}

return 0;
}