#pragma once
#include "backend/sycl/sycl_backend.h"


template<typename T> coordinate<T> sycl_backend<T>::mean_kinetic_momentum() const {
    coordinate<T> mean{};   // Sum of vi * mi;
    {
        auto x_reduction_buffer = sycl::buffer<T>(&mean.x(), 1U);
        auto y_reduction_buffer = sycl::buffer<T>(&mean.y(), 1U);
        auto z_reduction_buffer = sycl::buffer<T>(&mean.z(), 1U);
        q.submit([&](sycl::handler& cgh) {
#ifdef SYCL_IMPLEMENTATION_HIPSYCL
             auto reduction_x = sycl::reduction(x_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_y = sycl::reduction(y_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_z = sycl::reduction(z_reduction_buffer.get_access(cgh), sycl::plus<T>{});
#else
             auto reduction_x = sycl::reduction(x_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_y = sycl::reduction(y_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_z = sycl::reduction(z_reduction_buffer, cgh, sycl::plus<>{});
#endif
             cgh.parallel_for(compute_range_size(size_, max_reduction_size), reduction_x, reduction_y, reduction_z,   //
                              [size = size_, momentums = momentums_.get()](sycl::nd_item<1> it, auto& x, auto& y, auto& z) {
                                  const auto i = it.get_global_linear_id();
                                  if (i >= size) return;
                                  const auto momentum = momentums[i];
                                  x.combine(momentum.x());
                                  y.combine(momentum.y());
                                  z.combine(momentum.z());
                              });
         }).wait_and_throw();
    }
    return mean / momentums_.size();
}

template<typename T> coordinate<T> sycl_backend<T>::compute_error_lennard_jones() const {
    coordinate<T> mean{};   // Sum of vi * mi;
    {
        auto x_reduction_buffer = sycl::buffer<T>(&mean.x(), 1U);
        auto y_reduction_buffer = sycl::buffer<T>(&mean.y(), 1U);
        auto z_reduction_buffer = sycl::buffer<T>(&mean.z(), 1U);
        q.submit([&](sycl::handler& cgh) {
#ifdef SYCL_IMPLEMENTATION_HIPSYCL
             auto reduction_x = sycl::reduction(x_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_y = sycl::reduction(y_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_z = sycl::reduction(z_reduction_buffer.get_access(cgh), sycl::plus<T>{});
#else
             auto reduction_x = sycl::reduction(x_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_y = sycl::reduction(y_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_z = sycl::reduction(z_reduction_buffer, cgh, sycl::plus<>{});
#endif
             cgh.parallel_for(compute_range_size(size_, max_reduction_size), reduction_x, reduction_y, reduction_z,   //
                              [size = size_, forces = forces_.get()](sycl::nd_item<1> it, auto& x, auto& y, auto& z) {
                                  const auto i = it.get_global_linear_id();
                                  if (i >= size) return;
                                  const auto momentum = forces[i];
                                  x.combine(momentum.x());
                                  y.combine(momentum.y());
                                  z.combine(momentum.z());
                              });
         }).wait_and_throw();
    }
    return mean;
}


template<typename T> T sycl_backend<T>::get_momentums_squared_norm() const {
    T sum{};
    {
        auto sum_buffer = sycl::buffer<T>(&sum, 1U);
        q.submit([&](sycl::handler& cgh) {
#ifdef SYCL_IMPLEMENTATION_HIPSYCL
             auto reduction_sum = sycl::reduction(sum_buffer.get_access(cgh), sycl::plus<T>{});
#else
             auto reduction_sum = sycl::reduction(sum_buffer, cgh, sycl::plus<>{});
#endif
             cgh.parallel_for(compute_range_size(size_, max_reduction_size), reduction_sum, [momentums = momentums_.get(), size_ = size_](sycl::nd_item<1> it, auto& red) {
                 const auto i = it.get_global_linear_id();
                 if (i >= size_) return;
                 red.combine(sycl::dot(momentums[i], momentums[i]));
             });
         }).wait_and_throw();
    }
    return sum;
}

template<typename T> T sycl_backend<T>::reduce_energies() const {
    T sum{};
    {
        auto sum_buffer = sycl::buffer<T>(&sum, 1U);
        q.submit([&](sycl::handler& cgh) {
#ifdef SYCL_IMPLEMENTATION_HIPSYCL
             auto reduction_sum = sycl::reduction(sum_buffer.get_access(cgh), sycl::plus<T>{});
#else
             auto reduction_sum = sycl::reduction(sum_buffer, cgh, sycl::plus<>{});
#endif
             cgh.parallel_for(   //
                     compute_range_size(size_, max_reduction_size), reduction_sum, [energies = particule_energy_.get(), size_ = size_](sycl::nd_item<1> it, auto& red) {
                         const auto i = it.get_global_linear_id();
                         if (i >= size_) return;
                         red.combine(energies[i]);
                     });
         }).wait_and_throw();
    }
    return sum;
}
