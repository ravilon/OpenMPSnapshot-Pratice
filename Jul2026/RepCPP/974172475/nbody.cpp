/**
 *  @brief Demo app: N-Body simulation with Fork Union and OpenMP.
 *  @author Ash Vardanian
 *  @file nbody.cpp
 *
 *  To control the script, several environment variables are used:
 *
 *  - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
 *  - `NBODY_ITERATIONS` - number of iterations to run the simulation (default: 1000).
 *  - `NBODY_BACKEND` - backend to use for the simulation (default: `fork_union_static`).
 *  - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
 *
 *  The backends include: `fork_union_static`, `fork_union_dynamic`, `openmp_static`, and `openmp_dynamic`.
 *  To compile and run:
 *
 *  @code{.sh}
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release
 *  cmake --build build_release --config Release
 *  NBODY_COUNT=128 NBODY_THREADS=$(nproc) build_release/scripts/fork_union_nbody
 *  @endcode
 *
 *  The default profiling scheme is to 1M iterations for 128 particles on each backend:
 *
 *  @code{.sh}
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=openmp_static build_release/scripts/fork_union_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=openmp_dynamic build_release/scripts/fork_union_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=fork_union_static build_release/scripts/fork_union_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=fork_union_dynamic build_release/scripts/fork_union_nbody
 *  @endcode
 */
#include <vector> // `std::vector`
#include <random> // `std::random_device`, `std::uniform_real_distribution`
#include <thread> // `std::thread::hardware_concurrency`
#include <span>   // `std::span`
#include <bit>    // `std::bit_cast`

// Clang generally defines `_OPENMP` when OpenMP, but compiling it is
/// tricky and the header may not be available.
#if defined(_OPENMP)
#if __has_include(<omp.h>)
#include <omp.h>
#else
#undef _OPENMP
#endif
#endif

#include <fork_union.hpp>

namespace fu = ashvardanian::fork_union;

#if defined(__GNUC__) || defined(__clang__)
#define _FU_RESTRICT __restrict__
#else
#define _FU_RESTRICT
#endif

#pragma region - Shared Logic

static constexpr float g_const = 6.674e-11f;
static constexpr float dt_const = 0.01f;
static constexpr float softening_const = 1e-9f;

struct vector3_t {
    float x, y, z;

    inline vector3_t &operator+=(vector3_t const &other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

struct body_t {
    vector3_t position;
    vector3_t velocity;
    float mass;
};

inline float fast_rsqrt(float x) noexcept {
    std::uint32_t i = std::bit_cast<std::uint32_t>(x);
    i = 0x5f3759df - (i >> 1);
    float y = std::bit_cast<float>(i);
    float x2 = x * 0.5f;
    y = y * (1.5f - x2 * y * y);
    return y;
}

inline vector3_t gravitational_force(body_t const &bi, body_t const &bj) noexcept {
    float dx = bj.position.x - bi.position.x;
    float dy = bj.position.y - bi.position.y;
    float dz = bj.position.z - bi.position.z;
    float l2_squared = dx * dx + dy * dy + dz * dz + softening_const;
    float l2_reciprocal = fast_rsqrt(l2_squared);
    float l2_cube_reciprocal = l2_reciprocal * l2_reciprocal * l2_reciprocal;
    float mag = g_const * bi.mass * bj.mass * l2_cube_reciprocal;
    return {mag * dx, mag * dy, mag * dz};
}

inline void apply_force(body_t &bi, vector3_t const &f) noexcept {
    bi.velocity.x += f.x / bi.mass * dt_const;
    bi.velocity.y += f.y / bi.mass * dt_const;
    bi.velocity.z += f.z / bi.mass * dt_const;
    bi.position.x += bi.velocity.x * dt_const;
    bi.position.y += bi.velocity.y * dt_const;
    bi.position.z += bi.velocity.z * dt_const;
}

#pragma endregion - Shared Logic

#pragma region - Backends

void iteration_openmp_static(body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces, std::size_t n) noexcept {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    }
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
#endif
}

void iteration_openmp_dynamic(body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces, std::size_t n) noexcept {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
#endif
}

using pool_t = fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t>;

void iteration_fork_union_static(pool_t &pool, body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces,
                                 std::size_t n) noexcept {
    pool.for_n(n, [=](std::size_t i) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    });
    pool.for_n(n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

void iteration_fork_union_dynamic(pool_t &pool, body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces,
                                  std::size_t n) noexcept {
    pool.for_n_dynamic(n, [=](std::size_t i) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    });
    pool.for_n_dynamic(n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

#if FU_ENABLE_NUMA
using linux_numa_bodies_allocator_t = fu::linux_numa_allocator<body_t>;
using linux_numa_bodies_t = std::vector<body_t, linux_numa_bodies_allocator_t>;
using linux_distributed_pool_t = fu::linux_distributed_pool<fu::standard_yield_t>;

std::vector<linux_numa_bodies_t> make_buffers_for_fork_union_numa(linux_distributed_pool_t &pool,
                                                                  std::size_t n) noexcept {
    fu::numa_topology_t const &topology = pool.topology();
    std::size_t const numa_nodes_count = topology.nodes_count();

    std::vector<linux_numa_bodies_t> result;
    for (std::size_t i = 0; i < numa_nodes_count; ++i) {
        fu::numa_node_id_t const node_id = topology.node(i).node_id;
        linux_numa_bodies_allocator_t allocator(node_id);
        linux_numa_bodies_t bodies(n, allocator);
        result.emplace_back(std::move(bodies));
    }

    return result;
}

void iteration_fork_union_numa_static(linux_distributed_pool_t &pool, body_t *_FU_RESTRICT bodies,
                                      vector3_t *_FU_RESTRICT forces, std::size_t n,
                                      body_t **_FU_RESTRICT bodies_numa_copies) noexcept {

    using colocated_prong_t = typename linux_distributed_pool_t::prong_t;

    // This is a quadratic complexity all-to-all interaction, and it's not clear how
    // it can "shard" to take advantage of NUMA locality, especially for a small `n` world.
    // Still, at least we can replicate the body positions onto every node just once per iteration,
    // to reduce the number of remote accesses, even if they are cached.
    pool.for_threads([&](auto thread_index) noexcept {
        std::size_t const numa_node_index = pool.thread_colocation(thread_index);
        std::size_t const threads_next_to_numa_node = pool.threads_count(numa_node_index);
        std::size_t const thread_local_index = pool.thread_local_index(thread_index, numa_node_index);

        fu::indexed_split_t const n_split {n, threads_next_to_numa_node};
        fu::indexed_range_t const n_subrange = n_split[thread_local_index];
        std::memcpy(                                                //
            &bodies_numa_copies[numa_node_index][n_subrange.first], //
            &bodies[n_subrange.first],                              //
            n_subrange.count * sizeof(body_t));
    });

    pool.for_n(n, [&](colocated_prong_t prong) noexcept {
        std::size_t const numa_node_index = prong.colocation;
        body_t const *numa_bodies = bodies_numa_copies[numa_node_index];
        body_t const body_i = numa_bodies[prong.task];

        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(body_i, numa_bodies[j]);
        forces[prong.task] = f;
    });
    pool.for_n(n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

void iteration_fork_union_numa_dynamic(linux_distributed_pool_t &pool, body_t *_FU_RESTRICT bodies,
                                       vector3_t *_FU_RESTRICT forces, std::size_t n,
                                       body_t **_FU_RESTRICT bodies_numa_copies) noexcept {

    using colocated_prong_t = typename linux_distributed_pool_t::prong_t;

    // This expressions is same as in `iteration_fork_union_numa_static` static version:
    pool.for_threads([&](auto thread_index) noexcept {
        std::size_t const numa_node_index = pool.thread_colocation(thread_index);
        std::size_t const threads_next_to_numa_node = pool.threads_count(numa_node_index);
        std::size_t const thread_local_index = pool.thread_local_index(thread_index, numa_node_index);

        fu::indexed_split_t const n_split {n, threads_next_to_numa_node};
        fu::indexed_range_t const n_subrange = n_split[thread_local_index];
        std::memcpy(                                                //
            &bodies_numa_copies[numa_node_index][n_subrange.first], //
            &bodies[n_subrange.first],                              //
            n_subrange.count * sizeof(body_t));
    });

    // The rest only differs in the `for_n_dynamic` usage over `for_n`:
    pool.for_n_dynamic(n, [&](colocated_prong_t prong) noexcept {
        std::size_t const numa_node_index = prong.colocation;
        body_t const *numa_bodies = bodies_numa_copies[numa_node_index];
        body_t const body_i = numa_bodies[prong.task];

        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(body_i, numa_bodies[j]);
        forces[prong.task] = f;
    });
    pool.for_n_dynamic(n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

#endif // FU_ENABLE_NUMA

#pragma endregion - Backends

int main(void) {
    std::printf("Welcome to the Fork Union N-Body simulation!\n");

    // Read env vars
    auto const n_str = std::getenv("NBODY_COUNT");
    auto const iterations_str = std::getenv("NBODY_ITERATIONS");
    auto const backend_str = std::getenv("NBODY_BACKEND");
    auto const threads_str = std::getenv("NBODY_THREADS");

    // Parse env vars and validate
    std::size_t n = std::strtoull(n_str ? n_str : "0", nullptr, 10);
    std::size_t const iterations = std::strtoull(iterations_str ? iterations_str : "1000", nullptr, 10);
    std::string_view const backend = backend_str ? backend_str : "fork_union_static";
    std::size_t threads = std::strtoull(threads_str ? threads_str : "0", nullptr, 10);
    if (threads == 0) threads = std::thread::hardware_concurrency();
    if (n == 0) n = threads;

    // Prepare bodies and forces - 2 memory allocations
    std::vector<body_t> bodies(n);
    std::vector<vector3_t> forces(n);

    // Random generators are quite slow, but let's hope this doesn't take too long
    std::uniform_real_distribution<float> coordinate_distribution(0.f, 1.f);
    std::uniform_real_distribution<float> mass_distribution(1e20f, 1e25f);
    std::random_device random_device;
    std::mt19937 random_gen(random_device());
    for (std::size_t i = 0; i < n; ++i) {
        bodies[i].position.x = coordinate_distribution(random_gen);
        bodies[i].position.y = coordinate_distribution(random_gen);
        bodies[i].position.z = coordinate_distribution(random_gen);
        bodies[i].velocity.x = coordinate_distribution(random_gen);
        bodies[i].velocity.y = coordinate_distribution(random_gen);
        bodies[i].velocity.z = coordinate_distribution(random_gen);
        bodies[i].mass = mass_distribution(random_gen);
    }

#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
    if (backend == "openmp_static") {
        for (std::size_t i = 0; i < iterations; ++i) iteration_openmp_static(bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
    if (backend == "openmp_dynamic") {
        for (std::size_t i = 0; i < iterations; ++i) iteration_openmp_dynamic(bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
#endif

    // Every other configuration uses Fork Union
    pool_t pool;
    if (backend == "fork_union_static") {
        if (!pool.try_spawn(threads)) {
            std::fprintf(stderr, "Failed to spawn thread pool\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < iterations; ++i) //
            iteration_fork_union_static(pool, bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
    if (backend == "fork_union_dynamic") {
        if (!pool.try_spawn(threads)) {
            std::fprintf(stderr, "Failed to spawn thread pool\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < iterations; ++i)
            iteration_fork_union_dynamic(pool, bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }

#if FU_ENABLE_NUMA
    fu::numa_topology_t topology;
    if (!topology.try_harvest()) {
        std::fprintf(stderr, "Failed to harvest NUMA topology\n");
        return EXIT_FAILURE;
    }

    linux_distributed_pool_t numa_pool(std::move(topology));
    std::vector<linux_numa_bodies_t> bodies_numa_arrays = make_buffers_for_fork_union_numa(numa_pool, n);
    std::vector<body_t *> bodies_numa_buffers(bodies_numa_arrays.size());
    for (std::size_t i = 0; i < bodies_numa_arrays.size(); ++i) bodies_numa_buffers[i] = bodies_numa_arrays[i].data();

    if (backend == "fork_union_numa_static") {
        if (!numa_pool.try_spawn(threads)) {
            std::fprintf(stderr, "Failed to spawn NUMA thread pools\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < iterations; ++i)
            iteration_fork_union_numa_static(numa_pool, bodies.data(), forces.data(), n, bodies_numa_buffers.data());
        return EXIT_SUCCESS;
    }
    if (backend == "fork_union_numa_dynamic") {
        if (!numa_pool.try_spawn(threads)) {
            std::fprintf(stderr, "Failed to spawn NUMA thread pools\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < iterations; ++i)
            iteration_fork_union_numa_dynamic(numa_pool, bodies.data(), forces.data(), n, bodies_numa_buffers.data());
        return EXIT_SUCCESS;
    }
#endif // FU_ENABLE_NUMA

    std::fprintf(stderr, "Unsupported backend: %s\n", backend.data());
    return EXIT_FAILURE;
}
