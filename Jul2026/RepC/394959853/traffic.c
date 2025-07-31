#include <omp.h>  
#include<stdio.h>
#include <sched.h>
#include <stdbool.h>
#include<math.h>
#define _GNU_SOURCE
int fib(int n)   //recursive function to calculate fibonacci
{
if (n < 2) {
return 1;
}
return fib(n - 2) + fib(n - 1);
}
int fact(int n) //recursive function to calculate factorial
{
if (n < 2) {
return 1;
}
return n * fact(n - 1);
}

bool isPrime(int n)
{
// Corner case
if (n <= 1)
return false;

// Check from 2 to square root of n
for (int i = 2; i < n; i++)
if (n % i == 0)
return false;

return true;
}
int main () 
{

int n=0;
printf("Enter the value of traffic light number : ");
scanf("%d",&n);
int a[10] ;      //will store the fibonacci numbers
int b[10];              //this array will store the factorial of the numbers
int c[10];              //this will store the first n prime numbers
int i,k=3,d=0;
int x=0;
#pragma omp parallel sections
{
//section to calculate the fibonacci numbers
#pragma omp section
{
if()
#pragma omp parallel for schedule(guided, 1)
for (i = 0; i <n; i++) 
{
a[i]=fib(i);
//printf("Fib(%d): %d\n", i, fib(i));
}

}

//section to generate the factorial of the numbers
#pragma omp section
{
#pragma omp parallel for schedule(guided, 1)
for (i = 1; i <=n; i++) 
{
b[i-1]=fact(i);
//printf("Fact(%d): %d\n", i, fact(i));
}
}

//generate the first n prime numbers
#pragma omp section
{

#pragma omp parallel for schedule(guided, 1)
for (i = 1; i <=n*n; i++) 
{
if(isPrime(i))
{  
if(d<n)
{
c[d]=i;
d++;
} 

}


}
}

}
//printing the fibonacci
printf("The first %d fibonacci numbers are :",n);
for (i=0; i < n; i++) 
{
printf("%d ",a[i]);
}
printf("\n");

//printing the factorials
printf("The first %d factorials are :",n);
for (i=0; i < n; i++) 
{
printf("%d ",b[i]);
}
printf("\n");

//printing the prime numbers
printf("The first %d prime numbers are :",n);
for (i=0; i < n; i++) 
{
printf("%d ",c[i]);

}
printf("\n");
}
