#pragma once
#include "sycl_backend.h"
#include "sycl_queries.hpp"
#include <utility>

namespace sim {


template<typename T> static inline void prefetch_constant(const T* ptr) {
#if defined(__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
    if constexpr (sizeof(T*) == 8) {
        asm("prefetchu.L1 [%0];" : : "l"(ptr));
    } else {
        asm("prefetchu.L1 [%0];" : : "r"(ptr));
    }
#else
    (void) ptr;
#endif
}

static inline auto compute_range_size(size_t size, size_t work_group_size) {
    return sycl::nd_range<1>(work_group_size * ((size + work_group_size - 1) / work_group_size), work_group_size);
}

#include "sycl_backend_impl_reductions.hpp"


template<typename T, int n_sym> struct lennard_jones_kernel;

template<typename T, int n_sym>
static inline sycl::event update_lennard_jones_field_impl_sycl(                                                                               //
        sycl::queue& q, size_t size, const coordinate<T>* __restrict coordinates, coordinate<T>* __restrict forces, T* __restrict energies,   //
        const configuration<T>& config, sycl::event in_evt) {

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(std::move(in_evt));
        cgh.parallel_for<lennard_jones_kernel<T, n_sym>>(   //
                sycl::range(size), [size = size, L = config.L, coordinates = coordinates, r_star = config.r_star, r_cut = config.r_cut, forces = forces, energies = energies,
                                    use_cutoff = config.use_cutoff, epsilon_star = config.epsilon_star](sycl::item<1> item) {
                    const auto i = item.get_linear_id();
                    if (i >= size) return;
                    auto this_particule_energy = T{};
                    auto this_particule_force = coordinate<T>{};
                    const auto this_particule = coordinates[i];
#if defined(SYCL_IMPLEMENTATION_ONEAPI) || defined(SYCL_IMPLEMENTATION_ONEAPI)
                    static constexpr auto syms = get_symetries<n_sym>();
#else
                    const auto syms = get_symetries<n_sym>();
#endif
                    for (auto j = 0U; j < size; ++j) {
#pragma unroll
                        for (const auto& sym: syms) {   // If we put the get_symetries functions here, huge stack frame usage on CUDA backend of LLVM.
                            const coordinate<T> delta{sym.x() * L, sym.y() * L, sym.z() * L};
                            const coordinate<T> other_particule = delta + coordinates[j];
                            const T squared_distance = compute_squared_distance(this_particule, other_particule);
                            if (squared_distance < get_epsilon<T>() || (use_cutoff && squared_distance > integral_power<2>(r_cut))) continue;   //
                            //if (sycl::none_of_group(item.get_sub_group(), commit_result)) continue;
                            const T frac_pow_2 = r_star * r_star / (squared_distance + get_epsilon<T>());
                            const T frac_pow_6 = integral_power<3>(frac_pow_2);
                            const T force_prefactor = (frac_pow_6 - 1) * frac_pow_6 * frac_pow_2;
                            this_particule_force += (this_particule - other_particule) * force_prefactor;   //* static_cast<T>(commit_result);
                            this_particule_energy += (frac_pow_6 - 2) * frac_pow_6;                         //* static_cast<T>(commit_result);
                        }
                    }
                    forces[i] = this_particule_force * epsilon_star * 48;
                    energies[i] = 2 * epsilon_star * this_particule_energy;
                });
    });
}


template<typename T>
static inline sycl::event update_lennard_jones_field_dispatch_impl(                                                                           //
        sycl::queue& q, size_t size, const coordinate<T>* __restrict coordinates, coordinate<T>* __restrict forces, T* __restrict energies,   //
        const configuration<T>& config, sycl::event in_evt) {

    if (config.n_symetries == 1) {
        return update_lennard_jones_field_impl_sycl<T, 1>(q, size, coordinates, forces, energies, config, in_evt);
    } else if (config.n_symetries == 27) {
        return update_lennard_jones_field_impl_sycl<T, 27>(q, size, coordinates, forces, energies, config, in_evt);
    } else if (config.n_symetries == 125) {
        return update_lennard_jones_field_impl_sycl<T, 125>(q, size, coordinates, forces, energies, config, in_evt);
    } else {
        throw std::runtime_error("Unsupported");
    }
}


template<typename T> void sycl_backend<T>::update_lennard_jones_field(const configuration<T>& config) {
    update_lennard_jones_field_dispatch_impl(q, size_, coordinates_.get(), forces_.get(), particule_energy_.get(), config, sycl::event{}).wait();
}


template<typename T> void sycl_backend<T>::randinit_momentums(T min, T max) {
    std::generate(tmp_buf_.begin(), tmp_buf_.end(), [=]() {   //
        return coordinate<T>(internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max));
    });
    q.copy(tmp_buf_.data(), momentums_.get(), size_).wait();
}

template<typename T> void sycl_backend<T>::center_kinetic_momentums() {
    auto mean = mean_kinetic_momentum();
    q.parallel_for(sycl::range(size_), [mean = mean, size_ = size_, momentums = momentums_.get()](sycl::item<1> it) {
         const auto i = it.get_linear_id();
         momentums[i] -= mean;
     }).wait();
}

template<typename T> void sycl_backend<T>::apply_multiplicative_correction_to_momentums(T coeff) {
    q.parallel_for(sycl::range(size_), [coeff = coeff, size_ = size_, momentums = momentums_.get()](sycl::item<1> it) {
         const auto i = it.get_linear_id();
         momentums[i] *= coeff;
     }).wait();
}

template<typename T> void sycl_backend<T>::store_particles_coordinates(pdb_writer& writer, size_t i, T temp, T epot) const {
    q.copy(coordinates_.get(), tmp_buf_.data(), size_).wait();
    writer.store_new_iter(i, tmp_buf_, temp, epot);
}


template<typename T> void sycl_backend<T>::run_velocity_verlet(const configuration<T>& config) {
    // First step: half step update of the momentums.
    auto evt = q.parallel_for(   //
            sycl::range(size_), [size = size_, momentums = momentums_.get(), forces = forces_.get(), conversion_force = config.conversion_force, dt = config.dt](sycl::item<1> it) {
                const auto i = it.get_linear_id();
                momentums[i] += conversion_force * forces[i] * dt / 2;
            });

    // Second step: update particules positions
    auto evt2 = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(evt);
        cgh.parallel_for(sycl::range(size_), [size = size_, coordinates = coordinates_.get(), momentums = momentums_.get(), m_i = config.m_i, dt = config.dt](sycl::item<1> it) {
            const auto i = it.get_linear_id();
            coordinates[i] += dt * momentums[i] / m_i;
        });
    });

    auto evt3 = update_lennard_jones_field_dispatch_impl(q, size_, coordinates_.get(), forces_.get(), particule_energy_.get(), config, evt2);

    // Last step: update momentums given new forces
    auto evt4 = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(evt3);
        cgh.parallel_for(sycl::range(size_),
                         [size = size_, momentums = momentums_.get(), forces = forces_.get(), conversion_force = config.conversion_force, dt = config.dt](sycl::item<1> it) {
                             const auto i = it.get_linear_id();
                             momentums[i] += conversion_force * forces[i] * dt / 2;
                         });
    });

    auto evt5 = update_lennard_jones_field_dispatch_impl(q, size_, coordinates_.get(), forces_.get(), particule_energy_.get(), config, evt4);
    evt5.wait_and_throw();
}

template<typename T>
EXPORT sycl_backend<T>::sycl_backend(size_t size, const sycl::queue& queue)
    : q(queue), size_(size), coordinates_(size, q), momentums_(size, q), forces_(size, q), particule_energy_(size, q), tmp_buf_(size) {

    max_reduction_size = 256U;

#ifdef SYCL_IMPLEMENTATION_ONEAPI
    if (q.get_device().is_cpu()) {
        max_reduction_size = std::min(64UL, max_reduction_size);
    } else if (q.get_device().is_gpu()) {
        max_reduction_size = std::min(512UL, max_reduction_size);
    }
#endif
}

template<typename T> void sycl_backend<T>::set_particles(const std::vector<coordinate<T>>& particules, const configuration<T>&) {
    q.copy(particules.data(), coordinates_.get(), size_).wait();
}

template<typename T> void sycl_backend<T>::set_speeds(const std::vector<coordinate<T>>& speeds, const configuration<T>& config) {
    q.copy(speeds.data(), momentums_.get(), size_).wait();
    apply_multiplicative_correction_to_momentums(config.m_i);
}

template<typename T> std::tuple<coordinate<T>, T> sycl_backend<T>::get_last_lennard_jones_metrics() const { return {compute_error_lennard_jones(), reduce_energies()}; }

}   // namespace sim
