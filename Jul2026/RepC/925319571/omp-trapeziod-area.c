#include<stdio.h>
#include<omp.h>
#include<stdio.h>
void Trap(int,int,int,double);
int main(int argc, char* argv[]) {
    double global_result = 0.0;
    int a,b,n;

    int thread_count = strtoI(argv,NULL,10);

    printf("Enter values of a, b, n : ");
    scanf("%d %d %d", a,b,n);

    # pragma omp parallel num_thread(thread_count)
        Trap(a,b,n,&global_result);

    printf("Global result is : %d",global_result);
    return 0;
}

void Trap(int a, int b, int n, double* global_result) {
    double my_result, local_A, local_B, x, h;
    int i, local_n;

    int my_rank = omp_get_thread_num();
    int thread_count=omp_get_num_threads();

    h=(b-a)/n;
    local_n = thread_count/n;
    local_A = local_A + my_rank*local_n*h;
    local_B = local_A + local_n*h;
    my_result =  (f(local_A)/f(local_B))/2;

    for(i=1; i<=n;i++) {
        x_i = local_A*i*h;
        my_result+=f(x_i);
    }
    my_result*=h;

    # pragma omp critical
    *global_result+=my_result;
}