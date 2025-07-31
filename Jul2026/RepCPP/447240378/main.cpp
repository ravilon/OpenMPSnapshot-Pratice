#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <omp.h>

#define DTYPE double
#define SIZE_T unsigned long long

// a scan requires only a binary associative operator : i.e. (a op b) op c = a op 9b op c)
inline DTYPE op(DTYPE a, DTYPE b) {
    return a+b; // sum
    //return a>b?a:b; // max
    //return a<b?a:b; // min
}


const int TREE_LEVEL = 3;
const int NUM_THREADS = 6;
const SIZE_T N = (((SIZE_T)1)<<20);
timeval start, end;

// Sequential scan algorithm implementation
void get_groundtruth(std::vector<DTYPE>& gt, const std::vector<DTYPE>& in);

// Parallel scan algorithm implementations
void parallel_scan_1(const std::vector<DTYPE>* in, std::vector<DTYPE>* out);


// Helper functions
inline void print_vector(std::vector<DTYPE>& in);
inline float get_sec();
void check_result(std::vector<DTYPE>& gt, std::vector<DTYPE>& out);

int main(void) {

    /*******************************************
     * Data initialization
     * in : the input sequence.
     * out : the result of parallel algorithm
     * gt : the ground truth, calculated by the get_groundtruth() function
     ********************************************/
    std::cout << "************************************************************************\n";
    std::cout << "Parallel Scan Algorithm Start\n";
    std::cout << "--- The length of sequence is " << N <<"\n";
    std::cout << "--- The total memory size of sequence is " << N*sizeof(DTYPE)/1024.0/1024/1024 <<" GB \n";
    std::cout << "************************************************************************\n";
    srand(time(NULL));
    std::vector<DTYPE> in(N); 
    std::vector<DTYPE> out(N, 0); 
    std::generate(in.begin(), in.end(), [](){return std::rand()%20-10;});
    std::vector<DTYPE> gt(N, 0);


    /*******************************************
     * Getting the ground truth with a naive sequential algorithm
     ********************************************/
    std::cout << "\nSequential Scan Algorithm Start\n";
    gettimeofday(&start, NULL);
    get_groundtruth(gt, in);
    gettimeofday(&end, NULL);
    std::cout << "--- Total elapsed time : " << get_sec() << " s\n\n";


    /*******************************************
     * Getting the result of scan with parallel algorithm
     ********************************************/
    std::cout << "\nParallel Scan Algorithm Start\n";
    gettimeofday(&start, NULL);
    parallel_scan_1(&in, &out);
    gettimeofday(&end, NULL);
    std::cout << "--- Total elapsed time : " << get_sec() << " s\n\n";

    

    /*******************************************
     * Checking the correctness
     ********************************************/
    check_result(gt, out);
    return 0;

}

DTYPE scan_1_first_path(const std::vector<DTYPE>* in, std::vector<DTYPE>* out, SIZE_T start, SIZE_T end) {

    if (end-start <= (N>>TREE_LEVEL)) {
        
        ((*out)[start]) = ((*in)[start]);
        for (SIZE_T i=start+1; i<end; i++)
            ((*out)[i]) = op(((*out)[i-1]) , ((*in)[i]));
          

        return ((*out)[end-1]);
    }

    SIZE_T mid = (start+end)/2;
    DTYPE left=0, right=0;

    #pragma omp task shared(left)
    {
        left = scan_1_first_path(in, out, start, mid);
    }
    right = scan_1_first_path(in, out, mid, end);

    #pragma omp taskwait 
    ((*out)[end-1]) = op(left, right);
    return ((*out)[end-1]);
}

void scan_1_second_path(const std::vector<DTYPE>* in, std::vector<DTYPE>* out, SIZE_T start, SIZE_T end, DTYPE left) {

    if (end-start <= (N>>TREE_LEVEL)) {
        
        for (SIZE_T i=start; i<end-1; i++)
            ((*out)[i]) = op(((*out)[i]), left);
        ((*out)[end-1]) = op(((*out)[end-2]), ((*in)[end-1]));
        
        return;
    }

    SIZE_T mid = (start+end)/2;
    DTYPE temp = ((*out)[mid-1]);

    #pragma omp task
    {
        scan_1_second_path(in, out, start, mid, left);
    }
    scan_1_second_path(in, out, mid, end, op(left,temp));

    #pragma omp taskwait

}

void parallel_scan_1(const std::vector<DTYPE>* in, std::vector<DTYPE>* out) {

    /*
     * The parallel algorithm: Two-pass algorithm
     * - first path: get partial result
     * - second path: add all the left partial results to current value
     * Work : O(N), Span : O(logN)
     */

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp master
        {
            scan_1_first_path(in, out, 0, in->size());
            scan_1_second_path(in, out, 0, in->size(), 0);
        }
    }
}




void get_groundtruth(std::vector<DTYPE>& gt, const std::vector<DTYPE>& in) {
    /*
     * The simple sequenTal algorithm: accumulate the result from left to right
     * Work : O(N), Span : O(N)
     */
    gt[0] = in[0];
    for (SIZE_T i=1; i<gt.size(); i++) {
        gt[i] = op(gt[i-1], in[i]);
    }
    
}

void print_vector(std::vector<DTYPE>& in) {
    for(auto i=in.begin(); i!=in.end(); i++)
        std::cout << *i << " ";
    std::cout << std::endl;
}

float get_sec() {
    return (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)*1e-6;
}

void check_result(std::vector<DTYPE>& gt, std::vector<DTYPE>& out) {

    std::cout << "Start Checking result...\n";
    bool result = true;
    for (SIZE_T i=0; i<gt.size(); i++) {
        if (gt[i] != out[i]) {
            result = false;
            std::cout << "--- [[ERR]] Incorrect at out["<<i<<"] with result="<<out[i]<<" but, gt="<<gt[i]<<"\n";
            break;
        }
    }

    if (result) {
        std::cout << "--- The result is correct!!!\n";
    } else {
        std::cout << "--- The result is incorect!!!\n";
    }

}


