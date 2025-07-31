#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

bool isPrime(int n)
{
    // Corner case
    if (n <= 1)
        return false;
 
    // Check from 2 to square root of n
    #pragma omp parallel for
    for (int i = 2; i <= sqrt(n); i++)
        if (n % i == 0)
            return false;
 
    return true;
}

using namespace std;

int main() {

        auto startSerial = chrono::high_resolution_clock::now();
    int count = 0;
    for(int i=1;i<100;i+=2){
        if(isPrime(i) && isPrime(i+1)){
            count++;
        }
    }
        auto endSerial = chrono::high_resolution_clock::now();
            chrono::duration<double> durationSerial = endSerial - startSerial;
    double executionTimeSerial = durationSerial.count();

    cout<<"Number of prime numbers are : "<<count<<endl;
    cout<<"Time taken for execution is : "<<executionTimeSerial<<endl;

    return 0;
}

