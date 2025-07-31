#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <random>
using namespace std;
void parallel_pi(int numstep)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)numstep;
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) 
    for (i = 0; i < numstep; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();
            cout<<"-----------------For CLASSICAL PARALLEL : -----------------------------"<<endl;
                    cout<<"DHIVYESH R K 2021BCS0084"<<endl;
        cout<<"Numsteps : "<<numstep<<endl;
    cout << "pi = " << pi << std::endl;
    cout << "time = " << end - start << std::endl;
}
void serial_pi(int numstep)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)numstep;
    double start = omp_get_wtime();
//    #pragma omp parallel for private(x) 
    for (i = 0; i < numstep; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();

            cout<<"-----------------For SERIAL: -----------------------------"<<endl;
                    cout<<"DHIVYESH R K 2021BCS0084"<<endl;
        cout<<"Numsteps : "<<numstep<<endl;
    cout << "pi = " << pi << std::endl;
    cout << "time = " << end - start << std::endl;
}
void critical_pi(int numstep)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)numstep;
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) 
    for (i = 0; i < numstep; i++) {
        x = (i + 0.5) * step;
        #pragma omp critical
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();
        cout<<"-----------------For CRITICAL : -----------------------------"<<endl;
                cout<<"Numsteps : "<<numstep<<endl;
        cout<<"DHIVYESH R K 2021BCS0084"<<endl;

    cout << "pi = " << pi << std::endl;
    cout << "time = " << end - start << std::endl;
}
void atomic_pi(int numstep)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)numstep;
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) 
    for (i = 0; i < numstep; i++) {
        x = (i + 0.5) * step;
        #pragma omp atomic
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();
        cout<<"-----------------For ATOMIC : -----------------------------"<<endl;
                cout<<"DHIVYESH R K 2021BCS0084"<<endl;
        cout<<"Numsteps : "<<numstep<<endl;
    cout << "pi = " << pi << std::endl;
    cout << "time = " << end - start << std::endl;
}
void reduction_pi(int numstep)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)numstep;
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) reduction(+:sum)
    for (i = 0; i < numstep; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();

        cout<<"-----------------USING REDUCTION : -----------------------------"<<endl;
                cout<<"DHIVYESH R K 2021BCS0084"<<endl;
        cout<<"Numsteps : "<<numstep<<endl;
    cout << "pi = " << pi << std::endl;
    cout << "time = " << end - start << std::endl;
}

int main(){
	int numstep = 200;
	
	cout<<"200 NUMSTEPS : "<<endl;
	parallel_pi(numstep);
		serial_pi(numstep);
			atomic_pi(numstep);
				reduction_pi(numstep);
					critical_pi(numstep);

	cout<<"--------------------------------------------------"<<endl;
		
	numstep = 1000;
	
	cout<<"1000 NUMSTEPS : "<<endl;
	parallel_pi(numstep);
		serial_pi(numstep);
			atomic_pi(numstep);
				reduction_pi(numstep);
					critical_pi(numstep);

	cout<<"--------------------------------------------------"<<endl;
	numstep = 10000;
	
	cout<<"10000 NUMSTEPS : "<<endl;
	parallel_pi(numstep);
		serial_pi(numstep);
			atomic_pi(numstep);
				reduction_pi(numstep);
					critical_pi(numstep);

	cout<<"--------------------------------------------------"<<endl;
	numstep = 100000;
	
	cout<<"100000 NUMSTEPS : "<<endl;
	parallel_pi(numstep);
		serial_pi(numstep);
			atomic_pi(numstep);
				reduction_pi(numstep);
					critical_pi(numstep);

	cout<<"--------------------------------------------------"<<endl;
	
}

