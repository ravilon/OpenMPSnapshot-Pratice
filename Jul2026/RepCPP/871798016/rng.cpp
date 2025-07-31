#include "rng.hpp"
#include <random>
#include <ctime>
#include <omp.h>

std::vector<float> rng::float_vector(const size_t &size, const float &min, const float &max, const int& t_num) {
    std::vector<float> data(size, 0.0f);
    float *data_ptr = data.data();

    std::seed_seq seq{time(nullptr)};
    std::vector<uint32_t> seeds(omp_get_max_threads(), 0);
    seq.generate(seeds.begin(), seeds.end());

    omp_set_num_threads(t_num);

    #pragma omp parallel
    {
        std::mt19937 mt(seeds[omp_get_thread_num()]);
        std::uniform_real_distribution<float> dist(min, max);

        #pragma omp for
        for (int i = 0; i < size; ++i) {
            data_ptr[i] = dist(mt);
        }  
    }

    return data;
}
