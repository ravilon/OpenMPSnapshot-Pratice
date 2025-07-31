#include <stdio.h>
#include <omp.h>

int main() 
{
    const int SIZE = 10;

    int a[SIZE] = {12, 6, 98, 234, 4, 56, 7, 800, 90, 0};
    int b[SIZE] = {1, 56, 52, 77, 921, 6, 98, 2, 5, 11}; 
 
    float av_a = 0;
    float av_b = 0;

    #pragma omp parallel for reduction(+: av_a, av_b)
    for (int i = 0; i < SIZE; i++)
    {
        av_a += a[i];
        av_b += b[i];
    }

    #pragma omp sections
    {
        #pragma omp section
        {
            av_a = av_a / SIZE;
        }
        #pragma omp section
        {
            av_b = av_b / SIZE;
        }
    }

    printf("Average for a[]: %.3f, for b[]: %.3f\n", av_a, av_b);
    if (av_a > av_b) {
        printf("Average value of a[] is greater than average of b[].\n");
    } else {
        printf("Average value of b[] is greater than average of a[].\n");
    }

    return 0;
}