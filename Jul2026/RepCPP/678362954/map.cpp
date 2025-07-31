#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <vector>

#ifndef TYPE
#define TYPE double
#endif

#ifndef OPT_BLOCK_SIZE
#define OPT_BLOCK_SIZE 32
#endif

#ifndef ALIGNMENT
#define ALIGNMENT 32
#endif


#include "../include/timer.hpp"
#include "../include/utils.hpp"

void block_sparse_matrix(TYPE* m1, int size, int stride, int input){

    assert((size%stride) == 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            auto m = i*size+j;
            if ((m%stride)==0 && ((m/stride)%2)==0 )
            {   
                for (size_t k = 0; k < stride; i++)
                {
                    m1[m+k] = input;
                }   
            }       
        }
    }
}

void range_usm_matrix_addition(sycl::queue &Q, int size, int dim, int iter, bool print){

    timer time;

    auto N = static_cast<size_t>(size);
    
    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size*size, Q);
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size*size, Q);
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size*size, Q); Q.wait();
    
    std::fill(m1,m1+(size*size*size),1.0); Q.wait();
    std::fill(m2,m2+(size*size*size),2.0); Q.wait();
    std::fill(m3,m3+(size*size*size),1.0); Q.wait();

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global(N);

        for (size_t i = 0; i < iter; i++)
        {
            time.start_timer();
            Q.submit([&](sycl::handler & cgh){
                cgh.parallel_for(sycl::range<1>(global),[=](sycl::item<1> it){

                    auto k = it.get_id(0);
                    auto N = it.get_range(0);
                    
                    for (size_t j = 0; j < N; j++)
                    {
                        for (size_t i = 0; i < N; i++)
                        {
                            m3[(k*N*N)+(j*N)+i] += m1[(k*N*N)+(j*N)+i] + m2[(k*N*N)+(j*N)+i];
                        }   
                    }  
                });
            });
            Q.wait();
            time.end_timer();
            timings[i] = time.duration();
        }
        
        if (print)
        {
            print_results(timings, iter, size, "mat-add(r)", dim, 5);
        }
    }
    else if (dim == 2)
    {
        sycl::range<2> global(N,N);

        for (size_t i = 0; i < iter; i++)
        {
            time.start_timer();
            Q.submit([&](sycl::handler & cgh){
                cgh.parallel_for(sycl::range<2>(global),[=](sycl::item<2> it){

                    auto k = it.get_id(0);
                    auto j = it.get_id(1);
                    auto N = it.get_range(0);
                    
                    for (size_t i = 0; i < N; i++)
                    {
                        m3[(k*N*N)+(j*N)+i] += m1[(k*N*N)+(j*N)+i] + m2[(k*N*N)+(j*N)+i];
                    }  
                });
            });
            Q.wait();
            time.end_timer();
            timings[i] = time.duration();
        }
        
        if (print)
        {
            print_results(timings, iter, size, "mat-add(r)", dim, 5);
        }

    }
    else if (dim == 3)
    {
        sycl::range<3> global(N,N,N);

        for (size_t i = 0; i < iter; i++)
        {
            time.start_timer();
            Q.submit([&](sycl::handler & cgh){
                cgh.parallel_for(sycl::range<3>(global),[=](sycl::item<3> it){

                    auto k = it.get_id(0);
                    auto j = it.get_id(1);
                    auto i = it.get_id(2);
                    auto N = it.get_range(0);

                    m3[(k*N*N)+(j*N)+i] += m1[(k*N*N)+(j*N)+i] + m2[(k*N*N)+(j*N)+i];
                });
            });
            Q.wait();
            time.end_timer();

            timings[i] = time.duration();
        }

        if (print)
        {
            print_results(timings, iter, size, "mat-add(r)", dim, 5);
        }
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 or 3 " << std::endl;
        sycl::free((TYPE*)m1,Q);
        sycl::free((TYPE*)m2,Q);
        sycl::free((TYPE*)m3,Q);
    }
    
    sycl::free((TYPE*)m1,Q);
    sycl::free((TYPE*)m2,Q);
    sycl::free((TYPE*)m3,Q);

}

void ndrange_usm_matrix_addition(sycl::queue &Q, int size, int dim, int block_size, int iter, bool print){

    timer time;

    auto N = static_cast<size_t>(size);
    auto N_b = static_cast<size_t>(block_size);

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size*size, Q);
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size*size, Q);
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size*size, Q); Q.wait();
    
    std::fill(m1,m1+(size*size*size),1.0);
    std::fill(m2,m2+(size*size*size),2.0);
    std::fill(m3,m3+(size*size*size),1.0);Q.wait();

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global(N);
        sycl::range<1> local(N_b);

        for (size_t i = 0; i < iter; i++)
        {
            time.start_timer();
            Q.submit([&](sycl::handler & cgh){
                cgh.parallel_for(sycl::nd_range<1>(global,local),[=](sycl::nd_item<1> it){

                    auto k = it.get_global_id(0);
                    auto N = it.get_global_range(0);
                    
                    for (size_t j = 0; j < N; j++)
                    {
                        for (size_t i = 0; i < N; i++)
                        {
                            m3[(k*N*N)+(j*N)+i] += m1[(k*N*N)+(j*N)+i] + m2[(k*N*N)+(j*N)+i];
                        }   
                    }  
                });
            });
            Q.wait();
            time.end_timer();;

            timings[i] = time.duration();
        }

        if (print)
        {
            print_results(timings, iter, size, "mat-add(ndr)", dim, 5);
        }
    }
    else if (dim == 2)
    {
        sycl::range<2> global(N,N);
        sycl::range<2> local(N_b,N_b);

        for (size_t i = 0; i < iter; i++)
        {
            time.start_timer();
            Q.submit([&](sycl::handler & cgh){
                cgh.parallel_for(sycl::nd_range<2>(global,local),[=](sycl::nd_item<2> it){

                    auto k = it.get_global_id(0);
                    auto j = it.get_global_id(1);
                    auto N = it.get_global_range(0);
                    
                    for (size_t i = 0; i < N; i++)
                    {
                        m3[(k*N*N)+(j*N)+i] += m1[(k*N*N)+(j*N)+i] + m2[(k*N*N)+(j*N)+i];
                    }  
                });
            });
            Q.wait();
            time.end_timer();
            timings[i] = time.duration();
        }
        
        if (print)
        {
            print_results(timings, iter, size, "mat-add(ndr)", dim, 5);
        }

    }
    else if (dim == 3)
    {
        sycl::range<3> global(N,N,N);
        sycl::range<3> local(N_b,N_b,N_b);

        for (size_t i = 0; i < iter; i++)
        {
            time.start_timer();
            Q.submit([&](sycl::handler & cgh){
                cgh.parallel_for(sycl::nd_range<3>(global,local),[=](sycl::nd_item<3> it){

                    auto k = it.get_global_id(0);
                    auto j = it.get_global_id(1);
                    auto i = it.get_global_id(2);
                    auto N = it.get_global_range(0);

                    m3[(k*N*N)+(j*N)+i] += m1[(k*N*N)+(j*N)+i] + m2[(k*N*N)+(j*N)+i];
                });
            });
            Q.wait();
            time.end_timer();
            timings[i] = time.duration();
        }
        
        if (print)
        {
            print_results(timings, iter, size, "mat-add(ndr)", dim, 5);
        }
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 or 3 " << std::endl;
        sycl::free((TYPE*)m1,Q);
        sycl::free((TYPE*)m2,Q);
        sycl::free((TYPE*)m3,Q);
    }
    
    sycl::free((TYPE*)m1,Q);
    sycl::free((TYPE*)m2,Q);
    sycl::free((TYPE*)m3,Q);

}

void range_usm_matrix_transpose(sycl::queue &Q, int size, int dim, int iter, bool print){

    timer time;

    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size, Q);
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size, Q);Q.wait();

    std::fill(m1,m1+(size*size),2.0); Q.wait();
    std::fill(m2,m2+(size*size),0.0); Q.wait();

    sycl::range<2> global(N,N);

    for (size_t i = 0; i < iter; i++)
    {
        time.start_timer();
        Q.submit([&](sycl::handler & cgh){
            cgh.parallel_for(sycl::range<2>(global),[=](sycl::item<2> it){

                auto k = it.get_id(0);
                auto j = it.get_id(1);
                auto N = it.get_range(0);
                
                auto temp =   m1[k*N+j];
                m2[j*N+k] = temp;
            });
        });
        Q.wait();
        time.end_timer();
        timings[i] = time.duration();
    }
    
    if (print)
    {
        print_results(timings, iter, size, "transpose(r)", dim, 5);
    }
    
    sycl::free((TYPE*)m1,Q);
    sycl::free((TYPE*)m2,Q);

}

void ndrange_usm_matrix_transpose(sycl::queue &Q, int size, int dim, int block_size, int iter, bool print){

    timer time;

    auto N = static_cast<size_t>(size);
    auto N_b = static_cast<size_t>(block_size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size, Q);
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(sizeof(TYPE)*size*size, Q);Q.wait();

    std::fill(m1,m1+(size*size),2.0); Q.wait();
    std::fill(m2,m2+(size*size),0.0); Q.wait();

    sycl::range<2> global(N,N);
    sycl::range<2> local{N_b,N_b};

    for (size_t i = 0; i < iter; i++)
    {
        time.start_timer();
        Q.submit([&](sycl::handler & cgh){
            cgh.parallel_for(sycl::nd_range<2>(global,local),[=](sycl::nd_item<2> it){

                auto k = it.get_global_id(0);
                auto j = it.get_global_id(1);
                auto N = it.get_global_range(0);
                
                auto temp =   m1[k*N+j];
                m2[j*N+k] = temp;
            });
        });
        Q.wait();
        time.end_timer();
        timings[i] = time.duration();
    }
    
    if (print)
    {
        print_results(timings, iter, size, "transpose(ndr)", dim, 5);
    }
    
    sycl::free((TYPE*)m1,Q);
    sycl::free((TYPE*)m2,Q);

}

void ndrange_usm_gemm_strided(sycl::queue &Q, int size, int block_size, const size_t batch){

    auto N = static_cast<size_t>(size);

    auto N_b = static_cast<size_t>(block_size);
    sycl::range<1> local{N_b};

    timer time;
#ifdef ALIGNED
#define STRIDE size*size
    TYPE * __restrict__ m1 = sycl::aligned_alloc_shared<TYPE>(ALIGNMENT,size*size*batch*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::aligned_alloc_shared<TYPE>(ALIGNMENT,size*size*batch*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::aligned_alloc_shared<TYPE>(ALIGNMENT,size*size*batch*sizeof(TYPE),Q); Q.wait();
#else
#define STRIDE size*size
    TYPE * __restrict__ m1 = sycl::malloc_shared<TYPE>(size*size*batch*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m2 = sycl::malloc_shared<TYPE>(size*size*batch*sizeof(TYPE),Q); Q.wait();
    TYPE * __restrict__ m3 = sycl::malloc_shared<TYPE>(size*size*batch*sizeof(TYPE),Q); Q.wait();
#endif

    std::fill(m1,m1+size*size*batch,1);
    std::fill(m2,m2+size*size*batch,1);
    std::fill(m3,m3+size*size*batch,0.0);

    //auto N_m = static_cast<size_t>(size*size);

    sycl::range<3> global1 {batch,N,N};
    sycl::range<3> local1{1, N_b,N_b};

    time.start_timer();
    Q.submit([&](sycl::handler & cgh){
        cgh.parallel_for(sycl::nd_range<3>(global1,local1),[=](sycl::nd_item<3> it){
            auto b = it.get_global_id(0);
            auto i = it.get_global_id(1);
            auto j = it.get_global_id(2);

            TYPE sum = 0.0;

            for (size_t k = 0; k < N; k++)
            {
                sum += m1[b*STRIDE+i*N+k]*m2[b*STRIDE+k*N+j];
            }
            m3[b*STRIDE+i*N+j]=sum;
        });
    });
    Q.wait();
    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Time taken : strided gemm with batch " << batch << " using nd_range( with usm ) "<< kernel_offload_time/(1E9) << " seconds\n" << std::endl;

    sycl::free(m1,Q);
    sycl::free(m2,Q);
    sycl::free(m3,Q);
}



