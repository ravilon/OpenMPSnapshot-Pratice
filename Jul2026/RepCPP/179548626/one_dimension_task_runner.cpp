//
// Created by ahmad on 4/8/19.
//

#include "one_dimension_task_runner.h"
#include <omp.h>

matrix_multiplication::task::OneDimensionTaskRunner::OneDimensionTaskRunner(size_t dataset_row, size_t dataset_col, int thread_num)
        : TaskRunner(dataset_row, dataset_col, thread_num) {
    dataset_.FillRandom(1, 100);
}

void matrix_multiplication::task::OneDimensionTaskRunner::RunParallel() {
    // Initializing partial result grabber
    auto partial_result = new matrix_multiplication::data::Matrix<int>[dataset_row_];
    matrix_multiplication::data::Matrix<int> result(dataset_row_, dataset_col_);
    result.Initialize(0);

    for (size_t i = 0; i < dataset_row_; ++i) {
        partial_result[i].SetDimensions(dataset_row_, dataset_col_);
    }

    // Initializing OpenMP Parallel Section
#pragma omp parallel num_threads(thread_num_)
    {
#pragma omp single
        for (size_t i = 0; i < dataset_row_; ++i) {
#pragma omp task
            partial_result[i] = dataset_.PartialMultiply(i, i + 1, 0, dataset_col_);
        }
    }

    for (size_t i = 0; i < dataset_row_; ++i) {
        result.Add(partial_result[i]);
        partial_result[i].dispose();
    }

    if (result == dataset_.Multiply())
        std::cout << "Successful result" << std::endl;
    else
        std::cout << "Failure result" << std::endl;

    delete[](partial_result);
}
