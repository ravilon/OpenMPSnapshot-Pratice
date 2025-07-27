#include <omp.h>
#define _GNU_SOURCE
#include<stdio.h>
#include <sched.h>
#include <string.h>

int main()
{
    int i;
    int n;
    printf("Enter the size: ");
    scanf("%d",&n);
    int a[n];
    int even[n];
    int odd[n];
    int sume;int sumo;
    memset(even,0, sizeof(even[0]));
    memset(odd,0, sizeof(odd[0]));
    printf("Enter the array elements: ");
    for(int k=0;k<n;k++)
    {
        scanf("%d",&a[k]);
    }
    #pragma omp parallel for schedule(static,2)  //can be changed to dynamic
    for(i=0;i<n;i++)
    {
        if(a[i]%2==0)
        {
            even[i]=a[i];
        }
        else
        {
            odd[i]=a[i];
        }
    }
    printf("Count of even numbers: ");
    for(int l=0;l<n;l++)
    {
        if(even[l]!=0)
            printf("%d \n",even[l]);
    }
    printf("Count of odd numbers: ");
    int k=0;
   
    for(int k=0;k<n;k++)
    {
        if(odd[k]!=0)
            printf("%d\n",odd[k]);
    }
    
    #pragma omp parallel for reduction(+:sume) schedule(static, 3)
    for (int p=0; p < n; p++)
    {
        sume += even[p];
    }
    printf("The sum of even numbers is %d\n",sume);
    #pragma omp parallel for reduction(+:sumo) schedule(static, 3)
    for (int p=0; p < n; p++)
    {
        sumo += odd[p];
    }
    printf("The sum of odd numbers is %d\n",sumo);
}