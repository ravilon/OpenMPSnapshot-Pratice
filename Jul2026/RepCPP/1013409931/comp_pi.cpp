#include <random>
#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>
#include <cstdlib>
#include <omp.h>

int main(int argc, char **argv) {
    int N,sum=0;
    if (argc > 1)
      N = atoi(argv[1]);
    else {
      N = 1000000;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    
    
    /*(x-0)+(y-0) = (r^2) is the equation for the circle in this case, with 0,0 as the origin.*/
    /*We calculate Pi with Monte-Carlo simulation.*/
    /*Is x^2 + y^2 <= r^2 ? If yes, then add it to the sum. The total number of iterations is N.*/
    #pragma omp parallel reduction (+:sum)
    {
      //Create random generator with unique seed
      unsigned int seed = omp_get_thread_num();
      std::default_random_engine generator(seed);
      //Create distribution - double values between -0.5 and 0.5
      std::uniform_real_distribution<double> distribution(-0.5,0.5);
      //Get random value
      #pragma omp for    
      for (int i = 0; i < N; i++){
        double x = distribution(generator);
        double y = distribution(generator);
          if (x*x + y*y < 0.25){
            sum++;
          }
        }
    }
    
    double pi = 4.0 * (sum/(double)N); /*Inside the circle cases divided by the number of iterations.*/
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Pi is approximately: " << pi << std::endl;
    std::cout << "took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";

    return 0;
}
