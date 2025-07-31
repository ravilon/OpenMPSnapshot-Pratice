#include <iostream>
#include <chrono>
#include <omp.h>



#define AVAL 3.0
#define BVAL 5.0
#define TOL  0.001


int main(){
    int Ndim, Pdim, Mdim;   /* A[N][P], B[P][M], C[N][M] */
    double cval, tmp, err, errsq, mflops, ORDER;
    int nthreads;

    std::cout<<"Enter The Order of the matrices:";
    std::cin>>ORDER;
    std::cout<<"Enter number of threads:";
    std::cin>>nthreads;

    omp_set_num_threads(nthreads);

    Ndim = ORDER;
	Pdim = ORDER;
	Mdim = ORDER;

    double* p_A = new double[Ndim*Pdim];
    double* p_B = new double[Pdim*Mdim];
    double* p_C = new double[Ndim*Mdim];

    
    #pragma omp parallel default(none) shared(Ndim, Pdim, Mdim, p_A, p_B, p_C)
    {
        // Intitalizing Matrix A
        #pragma omp for collapse(2)
        for(int i=0; i<Ndim;i++){
            for(int j=0; j<Pdim; j++){
                *(p_A+(i*Pdim+j)) = AVAL;
            }
        }

        // Intitalizing Matrix B
        #pragma omp for collapse(2)
        for(int i=0; i<Pdim;i++){
            for(int j=0; j<Mdim; j++){
                *(p_B+(i*Mdim+j)) = BVAL;
            }
        }

        // Intitalizing Matrix C
        #pragma omp for collapse(2)
        for(int i=0; i<Ndim;i++){
            for(int j=0; j<Mdim; j++){
                *(p_C+(i*Mdim+j)) = 0.0;
            }
        }  
    }
    auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel default(none) shared(Ndim, Pdim, Mdim, p_A, p_B, p_C)
    {
        #pragma omp for collapse(2) 
        for(int i=0; i<Ndim; i++){
            for(int j=0; j<Mdim; j++){
                double ele_sum = 0;
                for(int k=0; k<Pdim; k++){
                    // i-th row
                    ele_sum += *(p_A+(i*Pdim+k)) * *(p_B+(k*Pdim+j));
                }

                *(p_C+(i*Pdim+j)) = ele_sum;

            }
        }
    }
    

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dT = t1 - t0;

    std::cout<<" Order "<<ORDER<<" multiplication in "<<dT.count()<<" seconds"<<std::endl;

    mflops = (2.0 * (double)Ndim * (double)Pdim * (double)Mdim)/(1000000.0* dT.count());

    std::cout<<" Order "<<ORDER<<" multiplication at "<<mflops<<" mflops"<<std::endl;
    cval = Pdim * AVAL * BVAL;

    errsq = 0.0;

    #pragma omp parallel for collapse(2) default(none) shared(Ndim, Mdim, p_C, cval, errsq) private(err)
	for (int i=0; i<Ndim; i++){
		for (int j=0; j<Mdim; j++){
			err = *(p_C+i*Ndim+j) - cval;

            #pragma omp atomic
		    errsq += err * err;
		}
	}

    if (errsq > TOL) 
        std::cout<<"Errors in multiplication: "<<errsq<<std::endl;
	else
        std::cout<<"Hey, it worked"<<std::endl;

    std::cout<<"All Done"<<std::endl;

    delete[] p_A;
    delete[] p_B;
    delete[] p_C;

    return 0;
}