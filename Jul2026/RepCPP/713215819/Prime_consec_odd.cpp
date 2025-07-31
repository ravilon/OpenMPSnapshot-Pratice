#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <bits/stdc++.h>
using namespace std;


bool isPrime(int n)
{
    // Corner case
    if (n <= 1)
        return false;
    int end = 0;
    // Check from 2 to square root of n
    #pragma omp parallel for
    for (int i = 2; i < n ; i++)
        if (n % i == 0){
	      end = 1;
 	}
    if(end==0) return true;
    else return false;
}


int main() {

        auto startSerial = chrono::high_resolution_clock::now();
    int count = 0;
    cout<<"Done by DHIVYESH RK 2021BCS0084"<<endl;
    for(int i=1;i<100;i+=2){
        if(isPrime(i) && isPrime(i+2)){
            count+=1;
        }
    }
        auto endSerial = chrono::high_resolution_clock::now();
            chrono::duration<double> durationSerial = endSerial - startSerial;
    double executionTimeSerial = durationSerial.count();

    cout<<"Total  : "<<count<<endl;
    cout<<"Time taken for execution is : "<<executionTimeSerial<<endl;

    return 0;
}

