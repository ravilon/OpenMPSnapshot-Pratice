#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>
using namespace std;

#define TASK_SIZE 100

// class for records
class studentRecord
{
   public:
   int id;
   string name;
   float cgpa;
};

void print_list(studentRecord * x, int n) 
{
   	int i;
   	for (i = 0; i < n; i++) 
   	{
      	cout << x[i].id << " ";
      	cout << x[i].name << " ";
      	cout << x[i].cgpa << " ";
   		cout << "\n";
	}
}




/*************************************************MERGESORT USING NAME********************************************************/

void mergeSortNameAux(studentRecord *X, int n, studentRecord *tmp) 
{
   int i = 0;
   int j = n/2;
   int ti = 0;

   while (i<n/2 && j<n) 
   {
      if (X[i].name < X[j].name) 
	  {
         tmp[ti] = X[i];
         ti++; i++;
      } 
	  else 
	  {
         tmp[ti] = X[j];
         ti++; j++;
      }
   }
   while (i<n/2) 
   { 
      tmp[ti] = X[i];
      ti++; i++;
   }
   while (j<n) 
   { 
      tmp[ti] = X[j];
      ti++; j++;
   }
   
   for(int i=0;i<n;++i)
    {
         X[i]=tmp[i];
    }
} 

void mergeSortName(studentRecord *X, int n, studentRecord *tmp)
{
   printf("Thread id: %d doing %d\n",omp_get_thread_num(), n);
   if (n < 2) return;

   #pragma omp task shared(X) if (n > TASK_SIZE)
   {
   	 	mergeSortName(X, n/2, tmp);
   }
   #pragma omp task shared(X) if (n > TASK_SIZE)
   mergeSortName(X+(n/2), n-(n/2), tmp + n/2);

   #pragma omp taskwait
   mergeSortNameAux(X, n, tmp);
}

/************************************************************************************************************************************************/



/*************************************************MERGESORT USING ID********************************************************/

void mergeSortIDAux(studentRecord *X, int n, studentRecord *tmp) 
{
   int i = 0;
   int j = n/2;
   int ti = 0;

   while (i<n/2 && j<n) 
   {
      if (X[i].id < X[j].id)
	  {
         tmp[ti] = X[i];
         ti++; i++;
      } 
	  else 
	  {
         tmp[ti] = X[j];
         ti++; j++;
      }
   }
   while (i<n/2) 
   {
      tmp[ti] = X[i];
      ti++; i++;
   }
   while (j<n) 
   { 
      tmp[ti] = X[j];
      ti++; j++;
   }
   
   for(int i=0;i<n;++i)
    {
         X[i]=tmp[i];
    }
} 

void mergeSortID(studentRecord *X, int n, studentRecord *tmp)
{
  // printf("Thread id: %d doing %d\n",omp_get_thread_num(), n);
   if (n < 2) return;

   #pragma omp task shared(X) if (n > TASK_SIZE)
   {
   	 	mergeSortID(X, n/2, tmp);
   }
   #pragma omp task shared(X) if (n > TASK_SIZE)
   mergeSortID(X+(n/2), n-(n/2), tmp + n/2);

   #pragma omp taskwait
   mergeSortIDAux(X, n, tmp);
}

/************************************************************************************************************************************************/




/*************************************************MERGESORT USING CGPA********************************************************/

void mergeSortCGPAAux(studentRecord *X, int n, studentRecord *tmp) 
{
   int i = 0;
   int j = n/2;
   int ti = 0;

   while (i<n/2 && j<n) 
   {
      if (X[i].cgpa < X[j].cgpa) 
	  {
         tmp[ti] = X[i];
         ti++; i++;
      } 
	  else 
	  {
         tmp[ti] = X[j];
         ti++; j++;
      }
   }
   while (i<n/2) 
   { 
      tmp[ti] = X[i];
      ti++; i++;
   }
   while (j<n) 
   { 
      tmp[ti] = X[j];
      ti++; j++;
   }
   
   for(int i=0;i<n;++i)
    {
         X[i]=tmp[i];
    }
} 

void mergeSortCGPA(studentRecord *X, int n, studentRecord *tmp)
{
   printf("Thread id: %d doing %d\n",omp_get_thread_num(), n);
   if (n < 2) return;

   #pragma omp task shared(X) if (n > TASK_SIZE)
   {
   	 	mergeSortCGPA(X, n/2, tmp);
   }
   #pragma omp task shared(X) if (n > TASK_SIZE)
   mergeSortCGPA(X+(n/2), n-(n/2), tmp + n/2);

   #pragma omp taskwait
   mergeSortCGPAAux(X, n, tmp);
}

/************************************************************************************************************************************************/



int main() 
{
        int N  = 10000;
        int inpt;
        studentRecord X[N];
        studentRecord tmp[N]; 
        double start, stop;

        omp_set_dynamic(0);              /** Explicitly disable dynamic teams **/
        omp_set_num_threads(3);  
		
        int i=0;
        ifstream fin("input.txt");
        while(!fin.eof())
        {
           fin>>X[i].id>>X[i].name>>X[i].cgpa; //read data from the file student
           i++;
        }
        fin.close();


		cout << "List Before Sorting...\n";
	   	print_list(X, N);
	   	cout<<"\n";
	   	cout<<"1:sort by Name\n";
	   	cout<<"2:sort by ID\n";
	   	cout<<"3:sort by CGPA\n";
		cout<<"Please enter number:";
		cin>>inpt;
		
		// Switch case for option of mergesort type
		switch(inpt)
		{
				/********************SORT BY NAME CALL**********************/
			case 1:
	         start = omp_get_wtime();
	         #pragma omp parallel
	         {
	             #pragma omp single
	             mergeSortName(X, N, tmp);
	         }   
	         stop = omp_get_wtime();
       		 break;
       		
       			 /********************SORT BY ID CALL**********************/
			 case 2:   
	         start = omp_get_wtime();
	         #pragma omp parallel
	         {
	             #pragma omp single
	             mergeSortID(X, N, tmp);
	         }   
	         stop = omp_get_wtime();
	         break;
	         
	          /********************SORT BY CGPA CALL**********************/
	         case 3:
	         start = omp_get_wtime();
	         #pragma omp parallel
	         {
	             #pragma omp single
	             mergeSortCGPA(X, N, tmp);
	            
	         }   
	         stop = omp_get_wtime();
	         break;
		}

        print_list(X, N);
	 	printf("Time: %f (s) \n",stop-start);
	 
        free(X);
        free(tmp);
        return 0;
}
