#include <iostream>
#include <cmath>
#include <omp.h>
#include <time.h>
using namespace std;
//para ejecutar los distintos algoritmos, ir descomentando uno a uno
//por separado
void printArray(float *A, int size){
    for (int i = 0; i < size; i++)
        cout << A[i] << " ";
}
void merge(float *a, int l, int m, int r){
    int n1 = m - l + 1, n2 = r - m;
    float *L, *R;
    L=new float[n1];
    R=new float[n2];
    for (int i = 0; i < n1; i++){
        L[i] = a[l + i];
    }
    for (int j = 0; j < n2; j++) {
        R[j] = a[m + 1 + j];
    }
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            a[k] = L[i];
            i++;
        }
        else {
            a[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) {
        a[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        a[k] = R[j];
        j++;
        k++;
    }
}
void merge4(float *A,int l,int c,int d_c,int t_c,int r){
    merge(A,l,c,d_c);
    merge(A,d_c+1,t_c,r);
    merge(A,l,d_c,r);
}
void mergesort_serial(float *A, int l, int r){
    if(l<r){
        int m= floor((l+r)/2);
        mergesort_serial(A,l,m);
        mergesort_serial(A,m+1,r);
        merge(A,l,m,r);
    }
}

void mergesort4_serial(float *A,int l, int r){
    if(l<r){
        int m=floor((r-l)/4);
        mergesort4_serial(A,l,m+l);
        mergesort4_serial(A,(m+1)+l,(2*m)+l);
        mergesort4_serial(A,((2*m)+1)+l,(3*m)+l);
        mergesort4_serial(A,((3*m)+1)+l,r);
        merge4(A,l,m+l,(2*m)+l,(3*m)+l,r);
    }
}
void parallel_merge4sort(float *A, int l, int r){
    if(l<r) {
        int m = floor((r - l) / 4);
#pragma omp taskgroup
        {
#pragma omp task shared(A)
        mergesort4_serial(A, l, m + l);
#pragma omp task shared(A)
        mergesort4_serial(A, (m + 1) + l, (2 * m) + l);
#pragma omp task shared(A)
        mergesort4_serial(A, ((2 * m) + 1) + l, (3 * m) + l);
#pragma omp task shared(A)
        mergesort4_serial(A, ((3 * m) + 1) + l, r);
#pragma omp taskwait
        merge4(A, l, m + l, (2 * m) + l, (3 * m) + l, r);
    }
}
}
void parallel_mergesort(float *A, int l, int r){
    if(l<r){
        int m=(l+r)/2;
#pragma omp taskgroup
        {
#pragma omp task shared(A)
            parallel_mergesort(A, l, m);
#pragma omp task shared(A)
            parallel_mergesort(A, m + 1, r);
#pragma omp taskwait
            merge(A, l, m, r);
        }
    }
}
int main() {
    clock_t start,end;
    float time;
    int n;
    cout<<"Tamaño del arreglo: ";
    cin>>n;
    float* A;
    A=new float[n];
    //float A2[n];
    for(int i=0; i<n;i++){
        float r = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/2));
        A[i]=r;
        //A2[i]=r;
    }
    //float arr1[] = { 12, 11, 13, 5, 6, 7 };
    //float arr2[] = { 12, 11, 13, 5, 6, 7 };
    //int arr1_size = sizeof(arr1) / sizeof(arr1[0]);
    //int arr2_size = sizeof(arr2) / sizeof(arr2[0]);
/*    cout << "array \n";
    //printArray(A, n);

    //MERGE SORT
    start=clock();
    mergesort_serial(A, 0, n - 1);
    end=clock();
    time=(float)(end-start)/(CLOCKS_PER_SEC);
    cout << "\nmergesort: "<<time <<"s"<<endl;
    //printArray(A, n);
    */
/*
    //MERGE4SORT
    start=clock();
    mergesort4_serial(A, 0, n - 1);
    end=clock();
    time=(float)(end-start)/(CLOCKS_PER_SEC);
    cout << "\nmerge4sort: "<<time <<"s"<<endl;
    //printArray(A2, n);
    */

    //PARALLEL MERGESORT
 /*   omp_set_num_threads(24);
    #pragma omp parallel
    {
        #pragma omp single
        cout<<"threads: "<< omp_get_num_threads()<<endl;
        #pragma omp master
        start=clock();
        #pragma omp single
        parallel_mergesort(A,0,n-1);
        #pragma omp barrier
        end=clock();
    }
    time=(float)(end-start)/(CLOCKS_PER_SEC);
    cout<<"parallel mergesort: "<<time<<"s"<<endl;
*/
    //PARALLEL MERGE4SORT
    omp_set_num_threads(48);
    #pragma omp parallel
    {
        #pragma omp single
        cout<<"threads: "<< omp_get_num_threads()<<endl;
        #pragma omp master
        start=clock();
        #pragma omp single
        parallel_merge4sort(A,0,n-1);
        #pragma omp barrier
        end=clock();
    }
    time=(float)(end-start)/(CLOCKS_PER_SEC);
    cout<<"parallel merge4sort: "<<time<<"s"<<endl;

    return 0;
}