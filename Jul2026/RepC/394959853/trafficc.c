#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>
#include <unistd.h>
#define _GNU_SOURCE

//Decalring some global variables - 
bool l1=true,l2=true,l3=false,l4=true,l5=false,l6=true,l7=false,l8=true;

int main()
{
    int i;
    //Assuming a scenario with a total duration of 500 seconds
    #pragma omp parallel for schedule(static,2)
    for(i=0; i<1000; i++)
    {
        printf("\n");
        //We know that l1,l4,l6,l8 can continiously be switched on GREEN
        //Configuring for the other 4 traffic lights
        if (l2){
            l5 = l2;
            l3 = !(l2);
            l7 = !(l2);
        }

        if (l3){
            l7 = l3;
            l2 = !(l3);
            l5 = !(l3);
        }

        printf("Thread id is %d\n", omp_get_thread_num());
        //Printing the results - 
        printf("Traffic light 1 - GREEN \n");
        printf("Traffic light 4 - GREEN \n");
        printf("Traffic light 6 - GREEN \n");
        printf("Traffic light 8 - GREEN \n");

        if(l2)
            printf("Traffic light 2 - GREEN \n");
        else
            printf("Traffic light 2 - RED \n");


        if(l3)
            printf("Traffic light 3 - GREEN \n");
        else
            printf("Traffic light 3 - RED \n");

        if(l5)
            printf("Traffic light 5 - GREEN \n");
        else
            printf("Traffic light 5 - RED \n");

        if(l7)
            printf("Traffic light 7 - GREEN \n");
        else
            printf("Traffic light 7 - RED \n");

        // Allowing these light signals to stay same for 5 seconds before switch - 
        sleep(5);

        //Changing the 4 signals - 
        l2 = !l2;
        l3 = !l3;
        l5 = !l5;
        l7 = !l7;

        printf("\n");
        printf("\n");

        continue;
    }
}