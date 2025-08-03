#include <cmath>

#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
std::vector<float> result(input);
float sqrtdvadelpi = sqrt(2 / M_PI);

#pragma omp parallel for
for (int i = 0; i < result.size(); ++i) {
float x = result[i];
result[i] = 0.5f * x *
(1.0f + tanhf(sqrtdvadelpi * x * (1.0f + 0.044715f * x * x)));
}

return result;
}
