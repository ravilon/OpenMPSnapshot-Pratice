#inlcude <stdio.h>
#inlcude <stdlib.h>
#inlcude <omp.h>
int main(int argc, char *argv[]){
if (argc > 1){
omp_set_num_threads(atoi(argv[1]));
}
#pragma omp parallel 
{
printf("Hello, world\n");
}
return 0;
}