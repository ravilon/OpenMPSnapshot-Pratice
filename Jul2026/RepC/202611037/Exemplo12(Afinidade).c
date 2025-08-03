/*
- OMP_PLACES define os lugares aos quais as threads são atribuídas.

- OMP_PROC_BIND
•false: afinidade de thread desativada, o ambiente de execução pode mover encadeamentos entre locais.
•true: trava as threads ao núcleos.
•spread: distribui as threads igualmente entre os locais.
•close: empacota as threads perto da thread master na lista de locais.
•master: coloca as threads junto com a thread master.

export OMP_NUM_THREADS=2
export OMP_PLACES="threads|cores|sockets"
export OMP_PROC_BIND="false|true|spread|close|master"
*/

#include<stdio.h>
#include<stdlib.h>
#include <omp.h>

int main(int argc, int *argv[]){
#pragma omp parallel
while(1){

}
return 0;
}