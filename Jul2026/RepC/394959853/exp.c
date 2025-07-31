// #include <omp.h>  
// #include<stdio.h>
// #include <sched.h>
// #define _GNU_SOURCE
// int main () 
// {
//     int cube;
//     int a[]={0,1};
//     #pragma omp parallel for
//     for(int j=0;j<10;j++)
//     {
//         //printf("Thread: %d; ID: %d\n",omp_get_thread_num(),i);
//         a[j]=a[j-1]+a[j-2];
//         printf("Cube value = %d\n",a[j]);
//     } 

// }

#include <stdio.h>

long long fib(long long n) 
{
if (n < 2) {
return 1;
}
return fib(n - 2) + fib(n - 1);
}

int main(int argc, char ** argv) 
{
long long n = 0;

#pragma omp parallel for schedule(guided, 1)
for (n = 0; n <= 45; n++) {
printf("Fib(%lld): %lld\n", n, fib(n));
}

return 0;
}