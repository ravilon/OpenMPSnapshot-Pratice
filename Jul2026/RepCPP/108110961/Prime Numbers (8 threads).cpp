// Prime Numbers.cpp : Defines the entry point for the console application.
//sequential code 424 seconds 
//4 threads reduction dynamic 198 seconds
//8 threads reduction dynamic 184 seconds
#include "stdafx.h"
#include<omp.h>
#include<iostream>
#include<time.h>
#define Peak 500000
#define N 1

bool Is_Prime(int number)
{
bool isprime = true;
for (int denominator = (number - 1); denominator >= 2; denominator--)
{
//cout << denominator << "second loop" << endl;
if ((number%denominator) == 0)
{
isprime = false;
break;
}
if (isprime == false) break;
}
return isprime;
}
double Prime_Counter = 0;
bool IsPrime = true;

using namespace std;
int main()
{
double t1 = omp_get_wtime();
cout << "working" << endl;

#pragma omp parallel for num_threads(8) reduction(+:Prime_Counter) schedule(dynamic)
for (int Number = 2; Number < Peak; Number++)
{
IsPrime = Is_Prime(Number);
if (IsPrime)Prime_Counter++;

}

double t2 = omp_get_wtime();
cout << Prime_Counter << " prime numbers" << endl;
cout << t2 - t1 << "seconds"<<endl;

getchar();
return 0;
}

