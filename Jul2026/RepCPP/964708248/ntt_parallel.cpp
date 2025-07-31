#include "ntt_parallel.hpp"
#include "polynomial.hpp"
#include "evaluation.hpp"
#include "math.hpp"
#include "ntt.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <omp.h>

void ntt_cooley_tukey_iterative_parallel(polynomial_t *p, uint64_t r, uint64_t q, evaluation_t** res)
{
    // Number of bits required to bit-reverse.
    uint64_t nbits = __builtin_ctz(p->degree + 1);

    static uint64_t *roots;

    #pragma omp master
    {
        // Result evaluation. Must be shared and allocated once.
        *res = alloc_evaluation(p->degree + 1);
        // Root evaluations. Must be shared and allocated once.
        roots = static_cast<uint64_t *>(calloc(nbits, sizeof(uint64_t)));
    }
    #pragma omp barrier

    if (roots == nullptr)
    {
        fprintf(stderr, "Can't alloc memory for root precomputations at ntt_cooley_tukey_iterative!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize with bit-reversal positioning.
    #pragma omp for schedule(static)
    for (size_t i = 0; i < p->degree + 1; ++i)
    {
        (*res)->evaluations[i] = p->coefficients[bit_reverse(i, nbits)];
    }

    // Precompute generators required.
    #pragma omp for schedule(static)
    for (size_t i = 0; i < nbits; ++i)
    {
        uint64_t exponent = i == 0 ? 1 : 2 << (i - 1);
        roots[i] = modpow(r, exponent, q);
    }
    // Current generator used.
    size_t root_idx = nbits - 1;
    // Initial stride.
    uint64_t stride = 1;

    while (stride < p->degree + 1)
    {
        #pragma omp for schedule(static)
        for (size_t start = 0; start < p->degree + 1; start += stride * 2)
        {
            for (size_t i = start; i < start + stride; ++i)
            {
                // Butterfly
                uint64_t zp = modpow(roots[root_idx], i - start, q);
                uint64_t a = (*res)->evaluations[i];
                uint64_t b = (*res)->evaluations[i + stride];
                (*res)->evaluations[i] = modadd(a, modprod(zp, b, q), q);
                (*res)->evaluations[i + stride] = modsub(a, modprod(zp, b, q), q);
            }
        }


        stride <<= 1;
        root_idx -= 1;
    }

    #pragma omp master
    {
        free(roots);
        roots = nullptr;
    }
}

void intt_gentleman_sande_iterative_parallel(evaluation_t *e, uint64_t r, uint64_t q, polynomial_t** poly)
{

    // Number of bits required to bit-reverse.
    uint64_t nbits = __builtin_ctz(e->n);

    static uint64_t *roots;
    static uint64_t *coeff_copy;

    #pragma omp master
    {
        // Initialize polynomial. Shared for all threads.
        *poly = alloc_polynomial(e->n - 1);
        // Initialize roots. Shared for all threads.
        roots = static_cast<uint64_t *>(calloc(nbits, sizeof(uint64_t)));
        // Need a tmp copy to avoid overwrites.
        coeff_copy = static_cast<uint64_t *>(calloc(e->n, sizeof(uint64_t)));
    }
    #pragma omp barrier

    if (roots == nullptr)
    {
        fprintf(stderr, "Can't alloc memory for root precomputations at intt_gentleman_sande_iterative!\n");
        exit(EXIT_FAILURE);
    }

    if (coeff_copy == nullptr)
    {
        fprintf(stderr, "Can't alloc memory for coefficients copy at intt_gentleman_sande_iterative!\n");
        exit(EXIT_FAILURE);
    }

    // Precompute inverse
    r = modinv(r, q);

    // Initialize values.
    #pragma omp for schedule(static)
    for (size_t i = 0; i < e->n; ++i)
    {
        (*poly)->coefficients[i] = e->evaluations[i];
    }

    // Precompute inverse generators.
    #pragma omp for schedule(static)
    for (size_t i = 0; i < nbits; ++i)
    {
        uint64_t exponent = i == 0 ? 1 : 2 << (i - 1);
        roots[i] = modpow(r, exponent, q);
    }
    // Current root index.
    size_t root_idx = 0;
    // Initial stride.
    uint64_t stride = e->n / 2;

    while (stride > 0)
    {
        #pragma omp for schedule(static)
        for (size_t start = 0; start < e->n; start += stride * 2)
        {
            for (size_t i = start; i < start + stride; ++i)
            {
                // Gentleman-Sande butterfly
                uint64_t zp = modpow(roots[root_idx], i - start, q);
                uint64_t a = (*poly)->coefficients[i];
                uint64_t b = (*poly)->coefficients[i + stride];
                (*poly)->coefficients[i] = modadd(a, b, q);
                (*poly)->coefficients[i + stride] = modprod(modsub(a, b, q), zp, q);
            }
        }

        stride >>= 1;
        root_idx += 1;
    }

    uint64_t scaler = modinv(e->n, q);

    #pragma omp master
    {
        memcpy(coeff_copy, (*poly)->coefficients, e->n * sizeof(uint64_t));
    }
    #pragma omp barrier

    #pragma omp for schedule(static)
    for (size_t i = 0; i < e->n; ++i)
    {
        (*poly)->coefficients[i] = modprod(coeff_copy[bit_reverse(i, nbits)], scaler, q);
    }

    #pragma omp master
    {
        free(coeff_copy);
        free(roots);
        coeff_copy = nullptr;
        roots = nullptr;
    }
}

polynomial_t *multiplication_ntt_cyclic_convolution_auto_reduction_parallel(polynomial_t *p, polynomial_t *q, size_t d)
{
    omp_set_num_threads(num_threads);

    // Get the result degree.
    uint64_t res_degree = d - 1;

    // Assertions about the polynomial degrees.
    if (res_degree < p->degree || res_degree < q->degree)
    {
        fprintf(stderr, "Degree of result polynomial must be at least eaqul to the minimum degree of the given polynomials!\n");
        exit(EXIT_FAILURE);
    }

    // Set the degree of polynomials to the maximun d-1.
    change_degree(p, res_degree);
    change_degree(q, res_degree);

    // Get root generator and modulus.
    uint64_t mod = std::get<0>(parameters.at(res_degree));
    uint64_t root = std::get<1>(parameters.at(res_degree));

    evaluation_t *p_ntt = nullptr, *q_ntt = nullptr, *eval_res = nullptr;
    polynomial_t *res = nullptr;
    eval_res = alloc_evaluation(res_degree + 1);

    #pragma omp parallel shared(p_ntt, q_ntt, eval_res, res, root, mod, res_degree, p, q)
    {
        // Do NTT.
        ntt_cooley_tukey_iterative_parallel(p, root, mod, &p_ntt);
        //#pragma omp master
        ntt_cooley_tukey_iterative_parallel(q, root, mod, &q_ntt);

        // Wait till NTT done.
        #pragma omp barrier

        // Component-wise multiplication.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < res_degree + 1; ++i)
        {
            eval_res->evaluations[i] = modprod(p_ntt->evaluations[i], q_ntt->evaluations[i], mod);
        }

        // Get back polynomial result.
        intt_gentleman_sande_iterative_parallel(eval_res, root, mod, &res);

        // Wait till INTT done.
        #pragma omp barrier

        #pragma omp master
        {
            free_evaluation(eval_res);
            free_evaluation(q_ntt);
            free_evaluation(p_ntt);
        }
    }

    return res;
}

polynomial_t *multiplication_ntt_negative_cyclic_convolution_auto_reduction_parallel(polynomial_t *p, polynomial_t *q, size_t d)
{

    omp_set_num_threads(num_threads);

    // Get the result degree.
    uint64_t res_degree = d - 1;

    // Assertions about the polynomial degrees.
    if (res_degree < p->degree || res_degree < q->degree)
    {
        fprintf(stderr, "Degree of result polynomial must be at least eaqul to the minimum degree of the given polynomials!\n");
        exit(EXIT_FAILURE);
    }

    // Set the degree of polynomials to the maximun d-1.
    change_degree(p, res_degree);
    change_degree(q, res_degree);

    // Get root generators and modulus.
    uint64_t mod = std::get<0>(parameters.at(res_degree));
    uint64_t root = std::get<1>(parameters.at(res_degree));
    uint64_t root2 = std::get<2>(parameters.at(res_degree));

    // Precompute roots for preprocessing.
    uint64_t *root2_gen = static_cast<uint64_t *>(calloc(d, sizeof(uint64_t)));
    uint64_t *root2_gen_inv = static_cast<uint64_t *>(calloc(d, sizeof(uint64_t)));
    uint64_t root2_inv = modinv(root2, mod);

    evaluation_t *p_ntt = nullptr, *q_ntt = nullptr, *eval_res = nullptr;
    polynomial_t *res, *p_copy = nullptr, *q_copy = nullptr;
    eval_res = alloc_evaluation(res_degree + 1);

    #pragma omp parallel shared(p, q, p_ntt, q_ntt, eval_res, res, root2_gen, root2_gen_inv, p_copy, q_copy)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < d; ++i)
        {
            root2_gen[i] = modpow(root2, i, mod);
            root2_gen_inv[i] = modpow(root2_inv, i, mod);
        }

        // Preprocess p and q.
        #pragma omp master 
        {
            p_copy = alloc_polynomial(res_degree);
	        q_copy = alloc_polynomial(res_degree);
        }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (size_t i = 0; i <= res_degree; ++i)
        {
            p_copy->coefficients[i] = modprod(p->coefficients[i], root2_gen[i], mod);
            q_copy->coefficients[i] = modprod(q->coefficients[i], root2_gen[i], mod);
        }
        p = p_copy;
        q = q_copy;

        // Do NTT.
        ntt_cooley_tukey_iterative_parallel(p, root, mod, &p_ntt);
        ntt_cooley_tukey_iterative_parallel(q, root, mod, &q_ntt);

        // Component-wise multiplication.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < res_degree + 1; ++i)
        {
            eval_res->evaluations[i] = modprod(p_ntt->evaluations[i], q_ntt->evaluations[i], mod);
        }

        // Get back polynomial result.
        intt_gentleman_sande_iterative_parallel(eval_res, root, mod, &res);

        #pragma omp barrier

        // Restore the values.
        #pragma omp for schedule(static)
        for (size_t i = 0; i <= res_degree; ++i)
        {
            res->coefficients[i] = modprod(res->coefficients[i], root2_gen_inv[i], mod);
        }
    
        #pragma omp master
        {
            free_evaluation(eval_res);
            free_evaluation(p_ntt);
            free_evaluation(q_ntt);
            free_polynomial(p);
            free_polynomial(q);
        }
    }

    free(root2_gen);
    free(root2_gen_inv);

    return res;
}