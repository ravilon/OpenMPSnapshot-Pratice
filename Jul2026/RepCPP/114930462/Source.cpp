#include <omp.h>
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;

class Program
{
public:
static void Main(int num)
{
integralCalculation(function, 1, 4, 2, 3, 1250, num);
}

private:
static double function(double x, double y)
{
return y / x;
}

static void integralCalculation(double function(double, double), double a, double b, double c, double d, int n, int numThreads)
{
double startTime = omp_get_wtime();
srand(time(NULL));  
double result = 0;

#pragma omp parallel for shared(res) num_threads (numThreads)
for (int i = 0; i < n; i++)
{
double randomX = (double)rand() / RAND_MAX * (b - a) + a;  
double randomY = (double)rand() / RAND_MAX * (d - c) + c;
#pragma omp atomic  
result += function(randomX, randomY);
}
result *= ((b - a) * (d - c)) / n;
double time = (omp_get_wtime() - startTime) * 1000;
cout << "Integral = " << result << " Number of threads = ";
cout << numThreads << " Time = " << time << " milliseconds" << endl;
}
};

int main()
{
Program::Main(1);
Program::Main(2);
Program::Main(4);
Program::Main(8);
Program::Main(16);
}