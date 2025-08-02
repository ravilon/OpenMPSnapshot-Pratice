#include <omp.h>
#include<stdio.h>
#include <sched.h>
#define _GNU_SOURCE
int main()
{
    int i;
    int n;   
    //array containing first 20 natural numbers
    int a[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    printf("Enter the value of n: ");
    scanf("%d",&n);
    //stores the total sum
    int sum=0;
    n=n+2;
    #pragma omp parallel for schedule(static,2)
    for(i=2;i<n;i++)
    {
        int j=i;
        int total=i,k=2;
        while((j*k)<=20)
        {
            total+=j*k;    //calculates the factors of that number
            k++;
        }
        printf("Thread: %d; ID: %d\n",omp_get_thread_num(),i);
        printf("Total of factors of %d is: %d \n",i,total);
        sum+=total;
    }
    printf("The total sum of all %d numbers are: %d\n",(n-2),sum);
}