#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int max(int a, int b) { return (a > b) ? a : b; }


//FAILED!!
int knapSack(int W, int wt[], int val[], int n)
{
int x, y;
// Base Case
if (n == 0 || W == 0)
return 0;

#pragma omp parallel num_threads(4)
{
#pragma omp task
if (wt[n - 1] > W) y = knapSack(W, wt, val, n - 1);

#pragma omp task
if (wt[n - 1] <= W) x = max(val[n - 1] + knapSack(W - wt[n - 1], wt, val, n - 1),
knapSack(W, wt, val, n - 1));
}
return x;
}

int main()
{
//---------LEITURA DA ENTRADA
int n, W;

scanf("%d %d", &n, &W);
int *val = (int*) calloc(n, sizeof(int));
int *wt = (int*) calloc(n, sizeof(int));

int i;
for (i = 0; i < n; ++i) {
scanf("%d %d", &(val[i]), &(wt[i])); 
}
//--------------------------

printf("%d\n", knapSack(W, wt, val, n));
return 0;
}
