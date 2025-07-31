#include <omp.h>  
#include<stdio.h>
#include <sched.h>
#include <stdbool.h>
#include<math.h>
#define _GNU_SOURCE

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
printf("Enter the value of N : ");
scanf("%d",&n);
int c[10000];              //this will store the first n prime numbers
int i,k=3,d=0;
#pragma omp parallel sections
{

//generate the first n prime numbers
#pragma omp section
{

#pragma omp parallel for schedule(guided, 10) // this can be changed with dynamic and guided
for (i = 1; i <=n*n; i++) 
{
if(isPrime(i))
{  
if(d<n)
{
c[d]=i;
//printf("Prime : %d\n", i);
d++;
} 

}


}
}

}

//printing the prime numbers
printf("The first %d prime numbers are :",n);
for (i=0; i < n; i++) 
{
printf("%d ",c[i]);

}
printf("\n");
}

