///
/// \file src/WS/astroQUT.cpp
/// \brief AstroQUT solver implementation.
/// \author Jairo Diaz <jairo.diaz@unige.ch> 2016-2017
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#include "utils/linearop/operator/astrooperator.hpp"
#include "WS/astroQUT.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <vector>

#include <omp.h>

namespace alias
{
namespace WS
{


static Matrix<double> CenterOffset(std::string picture_path, int offset_vert, int offset_horiz, Parameters<double>& options)
{
#ifdef DEBUG
    std::cerr << "CenterOffset called" << std::endl;
#endif // DEBUG
    Matrix<double> result(options.pic_size, options.pic_size);
    Matrix<double> raw_picture(picture_path);
    size_t raw_pic_size = (size_t) sqrt(raw_picture.Length());
    size_t offset_base = (raw_pic_size-options.pic_size)/2;
    size_t offset_height = offset_base * (1 + offset_vert/100);
    size_t offset_width = offset_base * (1 + offset_horiz/100);
    #pragma omp parallel for simd collapse(2)
    for(size_t row = 0; row < options.pic_size; ++row)
        for(size_t col = 0; col < options.pic_size; ++col)
            result[row*options.pic_size + col] = raw_picture[(row+offset_height)*raw_pic_size + (col+offset_width)];

#ifdef DEBUG
    std::cerr << "CenterOffset done" << std::endl;
#endif // DEBUG
    return result;
}

static Matrix<double> Resample(Matrix<double> picture, size_t resample_windows_size)
{
#ifdef DEBUG
    std::cerr << "Resample called" << std::endl;
#endif // DEBUG
    // no resampling
    if(resample_windows_size == 1)
        return picture;

    Matrix<double> result(picture.Height(), picture.Width());
    std::random_device rnd;
    std::default_random_engine generator(rnd() + std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution uniform_dist(0,(int)(resample_windows_size*resample_windows_size-1));

    #pragma omp parallel for collapse(2)
    for(size_t block_row = 0; block_row < picture.Height(); block_row += resample_windows_size )
        for(size_t block_col = 0; block_col < picture.Width(); block_col += resample_windows_size )
        {
            // get all values from the block
            std::vector<double> block_values;
            for(size_t row = block_row; row < block_row + resample_windows_size; ++row)
                for(size_t col = block_col; col < block_col + resample_windows_size; ++col)
                    block_values.push_back(picture[row*picture.Width()+col]);

            // resample the block with replacement
            for(size_t row = block_row; row < block_row + resample_windows_size; ++row)
                for(size_t col = block_col; col < block_col + resample_windows_size; ++col)
                    result[row*picture.Width()+col] = block_values[uniform_dist(generator)];
        }
#ifdef DEBUG
    std::cerr << "Resample done" << std::endl;
#endif // DEBUG
    return result;
}

static Matrix<double> MCCompute(const Matrix<double>& mu_hat,
                                std::vector<std::poisson_distribution<int>>& mu_hat_dist,
                                std::default_random_engine generator)
{
#ifdef DEBUG
    std::cerr << "MCCompute called" << std::endl;
#endif // DEBUG
    Matrix<double> mu_hat_rnd(mu_hat.Height(), mu_hat.Width());

    #pragma omp simd
    for(size_t i = 0; i < mu_hat.Length(); ++i)
        mu_hat_rnd[i] = mu_hat_dist[i](generator);

    mu_hat_rnd -= mu_hat;
    mu_hat_rnd /= mu_hat;

#ifdef DEBUG
    std::cerr << "MCCompute done" << std::endl;
#endif // DEBUG
    return mu_hat_rnd;
}

static void BetaZero(const Matrix<double>& picture,
                     const AstroOperator<double>& astro,
                     Parameters<double>& options)
{
#ifdef DEBUG
    std::cerr << "BetaZero called" << std::endl;
#endif // DEBUG
    std::cout << "Computing beta0..." << std::endl;
    Matrix<double> x0(0.0, options.pic_size*2, 1);
    x0[0] = 1;
    Matrix<double> null_model = astro.BAW(x0, false, true, true, false);

    Matrix<double> non_zero_values(picture.Height(), picture.Width());
    size_t non_zero_values_amount = 0;

    for(size_t i = 0; i < picture.Length(); ++i)
        if( ! IsEqual(picture[i], 0.0) )
            non_zero_values[non_zero_values_amount++] = picture[i];

    // partial sort
    std::nth_element(&non_zero_values[0],
                     &non_zero_values[non_zero_values_amount/2],
                     &non_zero_values[non_zero_values_amount]);
    // get median
    double non_zero_values_median = non_zero_values[non_zero_values_amount/2];

    double null_model_sum = null_model.Sum();

    options.beta0 = non_zero_values_amount*non_zero_values_median/null_model_sum;
    std::cout << "beta0 = " << std::scientific << options.beta0 << std::endl << std::endl;
#ifdef DEBUG
    std::cerr << "BetaZero done" << std::endl;
#endif // DEBUG
}

static void Standardize(const Matrix<double>& mu_hat,
                        std::vector<std::poisson_distribution<int>>& mu_hat_dist,
                        const Matrix<double>& background,
                        const AstroOperator<double>& astro,
                        Parameters<double>& options)
{
#ifdef DEBUG
    std::cerr << "Standardize called" << std::endl;
#endif // DEBUG
    std::cout << "Computing standardization matrix..." << std::endl;
    Matrix<double> MC_astro(options.pic_size*2, options.MC_max);

    std::random_device rnd;
    #pragma omp parallel for schedule(dynamic)
    for(size_t MC_id = 0; MC_id < options.MC_max; ++MC_id)
    {
        std::default_random_engine generator(rnd() + omp_get_thread_num() + std::chrono::system_clock::now().time_since_epoch().count());
        if(options.MC_max > 100 && (MC_id % ((options.MC_max)/100) == 0))
            std::cout << "\r" + std::to_string(std::lround(MC_id*100.0/(double)(options.MC_max-1))) + "/100" << std::flush;

        Matrix<double> rnd_result = astro.WtAtBt(MCCompute(mu_hat, mu_hat_dist, generator), false, true, true, false).Abs();

        #pragma omp simd
        for(size_t i = 0; i < rnd_result.Height(); ++i )
            MC_astro[i*MC_astro.Width() + MC_id] = rnd_result[i];
    }
    std::cout << "\r100/100" << std::endl;

    options.standardize = Matrix<double>(1.0, options.model_size, 1);

    #pragma omp parallel for
    for(size_t i = 0; i < options.pic_size*2; ++i)
    {
        // partial sort
        std::nth_element(&MC_astro[i*options.MC_max],
                         &MC_astro[i*options.MC_max+options.MC_quantile_PF],
                         &MC_astro[(i+1)*options.MC_max]);
        // get quantile
        options.standardize[i] = MC_astro[i*options.MC_max+options.MC_quantile_PF];
    }

    // max rescale of wavelets and remove high frequency wavelets
    double wavelet_max_value = *std::max_element(&options.standardize[1], &options.standardize[options.pic_size]);
    #pragma omp parallel for simd
    for(size_t i = 0; i < options.pic_size/2; ++i)
        options.standardize[i] = wavelet_max_value;
    #pragma omp parallel for simd
    for(size_t i = options.pic_size/2; i < options.pic_size; ++i)
        options.standardize[i] = std::numeric_limits<double>::infinity();

    // spline ok

    // remove point sources from centre of the picture
    #pragma omp parallel for simd collapse(2)
    for(size_t i = 3*options.pic_size/8; i < 5*options.pic_size/8; ++i)
        for(size_t j = 3*options.pic_size/8; j < 5*options.pic_size/8; ++j)
            options.standardize[options.pic_size*2 + i*options.pic_size + j] = std::numeric_limits<double>::infinity();
    // remove point sources from empty background
    #pragma omp parallel for simd
    for(size_t i = 0; i < background.Length(); ++i)
        if( IsEqual(background[i], 0.0) )
        {
            options.standardize[options.pic_size*2 + i] = std::numeric_limits<double>::infinity();
#ifdef VERBOSE
            std::cout << i << " put to zero." << std::endl;
#endif // VERBOSE
        }
#ifdef DEBUG
    std::cerr << "Standardize done" << std::endl;
#endif // DEBUG
}

static void Lambda(const Matrix<double>& mu_hat,
                   std::vector<std::poisson_distribution<int>>& mu_hat_dist,
                   AstroOperator<double>& astro,
                   Parameters<double>& options)
{
#ifdef DEBUG
    std::cerr << "Lambda called" << std::endl;
#endif // DEBUG
    std::cout << "Computing lambda and lambdaI..." << std::endl;
    Matrix<double> lambda_standardize(options.standardize);
    lambda_standardize[0] = std::numeric_limits<double>::infinity();
    astro.Standardize(lambda_standardize);

    Matrix<double> WS_max_values(-std::numeric_limits<double>::infinity(), options.MC_max, 1);
    Matrix<double> PS_max_values(-std::numeric_limits<double>::infinity(), options.MC_max, 1);

    std::random_device rnd;
    #pragma omp parallel for schedule(dynamic)
    for(size_t MC_id = 0; MC_id < options.MC_max; ++MC_id)
    {

        std::default_random_engine generator(rnd() + omp_get_thread_num() + std::chrono::system_clock::now().time_since_epoch().count());

        if(options.MC_max > 100 && (MC_id % ((options.MC_max)/100) == 0))
            std::cout << "\r" + std::to_string(std::lround(MC_id*100.0/(double)(options.MC_max-1))) + "/100" << std::flush;

        Matrix<double> rnd_result = astro.WtAtBt(MCCompute(mu_hat, mu_hat_dist, generator)).Abs();

        // compute max value for each MC simulation in wavelet and spline results
        WS_max_values[MC_id] = *std::max_element(&rnd_result[0], &rnd_result[options.pic_size*2]);

        // compute max value for each MC simulation in point source results
        PS_max_values[MC_id] = *std::max_element(&rnd_result[options.pic_size*2], &rnd_result[options.model_size]);
    }
    std::cout << "\r100/100" << std::endl;

    // determine the value of lambda
    std::nth_element(&WS_max_values[0],
                     &WS_max_values[options.MC_quantile_PF],
                     &WS_max_values[options.MC_max]);
    options.lambda = WS_max_values[options.MC_quantile_PF];
    std::cout << "lambda = " << std::scientific << options.lambda << std::endl;

    // determine the value of lambdaI
    std::nth_element(&PS_max_values[0],
                     &PS_max_values[options.MC_quantile_PS],
                     &PS_max_values[options.MC_max]);
    double lambdaI = PS_max_values[options.MC_quantile_PS];
    std::cout << "lambdaI = " << std::scientific << lambdaI << std::endl;

    double PS_standardize_ratio = lambdaI/options.lambda;
    #pragma omp parallel for simd
    for(size_t i = options.pic_size*2; i < options.model_size; ++i)
        options.standardize[i] *= PS_standardize_ratio;
#ifdef DEBUG
    std::cerr << "Lambda done" << std::endl;
#endif // DEBUG
}

static void StandardizeAndRegularize(const Matrix<double>& background,
                                     AstroOperator<double>& astro,
                                     Parameters<double>& options)
{
#ifdef DEBUG
    std::cerr << "StandardizeAndRegularize called" << std::endl;
#endif // DEBUG
    std::cout << "Computing standardization and regularization values with ";
    std::cout << options.MC_max << " MC simulations..." << std::endl;
    Matrix<double> initial_guess(0.0, options.pic_size*2, 1);
    initial_guess[0] = 1.0;
    Matrix<double> u = astro.BAW(initial_guess, false, true, true, false);
    Matrix<double> mu_hat = background + u*options.beta0;
    std::vector<std::poisson_distribution<int>> mu_hat_dist(mu_hat.Length());
    #pragma omp parallel for simd
    for(size_t i = 0; i < mu_hat.Length(); ++i)
        mu_hat_dist[i] = std::poisson_distribution<int>(mu_hat[i]);

    astro.Transpose();

    Standardize(mu_hat, mu_hat_dist, background, astro, options);

    Lambda(mu_hat, mu_hat_dist, astro, options);

    astro.Transpose();
    astro.Standardize(options.standardize);
    std::cout << std::endl;
#ifdef DEBUG
    std::cerr << "StandardizeAndRegularize done" << std::endl;
#endif // DEBUG
}

static Matrix<double> Estimate(const Matrix<double>& picture,
                               const Matrix<double>& background,
                               const AstroOperator<double>& astro,
                               Parameters<double>& options)
{
#ifdef DEBUG
    std::cerr << "Estimate called" << std::endl;
#endif // DEBUG
    std::cout << "Computing static estimate..." << std::endl;
    options.fista_params.init_value = Matrix<double>(0.0, options.model_size, 1);
    options.fista_params.init_value[0] = options.beta0 * options.standardize[0];
    options.fista_params.indices = Matrix<size_t>(0,1+options.pic_size+options.pic_size*options.pic_size,1);
    #pragma omp parallel for simd
    for(size_t i = 1; i < 1+options.pic_size+options.pic_size*options.pic_size; ++i)
        options.fista_params.indices[i] = i + options.pic_size - 1;

    Matrix<double> result = fista::poisson::Solve(astro, background, picture, options.lambda, options.fista_params);

    result /= options.standardize;
    result.RemoveNeg(options.pic_size*2, options.model_size);
    Matrix<double> result_ps(&result[options.pic_size*2], options.pic_size, options.pic_size);
    Matrix<double> ps_cc_max = result_ps.ConnectedComponentsMax();
    result_ps.Data(nullptr);
    #pragma omp parallel for simd
    for(size_t i = 0; i < options.pic_size*options.pic_size; ++i)
        result[i + options.pic_size*2] = ps_cc_max[i];

    std::cout << std::endl;
#ifdef DEBUG
    std::cerr << "Estimate done" << std::endl;
#endif // DEBUG
    return result;
}

static Matrix<double> EstimateNonZero(const Matrix<double>& picture,
                                      const Matrix<double>& background,
                                      const Matrix<double>& solution_static,
                                      const AstroOperator<double>& astro,
                                      Parameters<double>& options)
{
#ifdef DEBUG
    std::cerr << "EstimateNonZero called" << std::endl;
#endif // DEBUG
    std::cout << "Getting non zero elements..." << std::endl;
    Matrix<size_t> non_zero_elements_indices = solution_static.NonZeroIndices();
    size_t non_zero_elements_amount = non_zero_elements_indices.Length();
    MatMult<double> non_zero_elements_operator(Matrix<double>(picture.Length(), non_zero_elements_amount), picture.Length(), non_zero_elements_amount);

    std::cout << "Generating non zero elements operator..." << std::endl;
    Matrix<double> identity(0.0, options.model_size, 1);
    for(size_t i = 0; i < non_zero_elements_amount; ++i)
    {
        if(non_zero_elements_amount > 100 && i % ((non_zero_elements_amount)/100) == 0)
            std::cout << "\r" + std::to_string(std::lround(i*100.0/(double)(non_zero_elements_amount-1))) + "/100" << std::flush;
        identity[non_zero_elements_indices[i]] = 1.0;
        Matrix<double> local_result = astro * identity;
        identity[non_zero_elements_indices[i]] = 0.0;
        #pragma omp parallel for simd
        for(size_t j = 0; j < local_result.Length(); ++j)
            non_zero_elements_operator[j*non_zero_elements_amount + i] = local_result[j];
    }
    std::cout << "\r100/100" << std::endl;

    options.fista_params.init_value = Matrix<double>(non_zero_elements_amount, 1);

    #pragma omp parallel for simd
    for(size_t i = 0; i < non_zero_elements_amount; ++i)
        options.fista_params.init_value[i] = solution_static[non_zero_elements_indices[i]] * options.standardize[non_zero_elements_indices[i]];

    size_t starting_index = 1;
    while(non_zero_elements_indices[starting_index] < options.pic_size)
        ++starting_index;
    Matrix<size_t> remove_neg_indices(non_zero_elements_amount-starting_index, 1);
    #pragma omp parallel for simd
    for(size_t i = 0; i < non_zero_elements_amount-starting_index; ++i)
        remove_neg_indices[i] = i + starting_index;
    options.fista_params.indices = remove_neg_indices;

    std::cout << "Generating the new beta estimates..." << std::endl;
    Matrix<double> beta_new = fista::poisson::Solve(non_zero_elements_operator,
                                                    background,
                                                    picture,
                                                    0.0,
                                                    options.fista_params);

    Matrix<double> result(solution_static);
    #pragma omp parallel for simd
    for(size_t i = 0; i < non_zero_elements_amount; ++i)
        result[non_zero_elements_indices[i]] = beta_new[i];
#ifdef DEBUG
    std::cerr << "EstimateNonZero done" << std::endl;
#endif // DEBUG
    return result;
}

static Matrix<double> SolveWS(const Matrix<double>& picture,
                              const Matrix<double>& sensitivity,
                              const Matrix<double>& background,
                              Parameters<double>& options)
{
#ifdef DEBUG
    std::cerr << "SolveWS called" << std::endl;
#endif // DEBUG
    AstroOperator<double> astro(options.pic_size, options.pic_size, options.pic_size/2, sensitivity, Matrix<double>(1, options.model_size, 1), false, options);

    BetaZero(picture, astro, options);

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    Matrix<double> solution;
    bool first = true;
    double prev_beta0 = std::numeric_limits<double>::infinity();
    size_t refine_max = 5;
    double total_time = 0;
    size_t iter_max = options.fista_params.iter_max;

    std::cout << "Computing wavelet and spline estimates." << std::endl;
    while( refine_max-- > 0 && std::abs(options.beta0-prev_beta0)/options.beta0 > 0.1 )
    {
        std::cout << "beta0 ratio = " << std::abs(options.beta0-prev_beta0)/options.beta0 << ". ";
        std::cout << "refines remaining: " << refine_max << ". ";
        std::cout << "Continuing..." << std::endl << std::endl;
        prev_beta0 = options.beta0;

        start = std::chrono::high_resolution_clock::now();
        StandardizeAndRegularize(background, astro, options);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_MC = end-start;

        if( !first )
        {
            #pragma omp parallel for simd
            for(size_t i = 0; i < options.pic_size; ++i)
                solution[i] += 1e-10;
            Matrix<size_t> zero_elements = solution.ZeroIndices();
            #pragma omp parallel for simd
            for(size_t i = 0; i < zero_elements.Length(); ++i)
                options.standardize[zero_elements[i]] = std::numeric_limits<double>::infinity();
            astro.Standardize(options.standardize);
            options.fista_params.iter_max = iter_max;
        }
        else
        {
            first = false;
            // first approximation should go further because of connected components step
            options.fista_params.iter_max = iter_max * 2;
        }

        start = std::chrono::high_resolution_clock::now();
        Matrix<double> solution_static = Estimate(picture, background, astro, options);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_FISTA = end-start;

        start = std::chrono::high_resolution_clock::now();
        solution = EstimateNonZero(picture, background, solution_static, astro, options);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_NZ = end-start;

        std::cout << std::defaultfloat << std::endl;
        std::cout << "Time for MC simulations: " << elapsed_time_MC.count() << " seconds" << std::endl;
        total_time += elapsed_time_MC.count();
        std::cout << "Time for FISTA solver: " << elapsed_time_FISTA.count() << " seconds" << std::endl;
        total_time += elapsed_time_FISTA.count();
        std::cout << "Time for GLM fit on non zero elements: " << elapsed_time_NZ.count() << " seconds" << std::endl << std::endl;
        total_time += elapsed_time_NZ.count();

        options.beta0 = solution[0]/options.standardize[0];
        std::cout << "New beta0 = " << options.beta0 << std::endl;
    }
    std::cout << "beta0 ratio = " << std::abs(options.beta0-prev_beta0)/options.beta0 << ". ";
    std::cout << "refines remaining: " << refine_max << ". ";
    std::cout << "Stopping." << std::endl << std::endl;

    std::cout << std::endl << "Total time: "   << total_time << " seconds" << std::endl << std::endl;
#ifdef DEBUG
    std::cerr << "SolveWS done" << std::endl;
#endif // DEBUG

    // restoring iter max to its original value
    options.fista_params.iter_max = iter_max;
    return solution/options.standardize;
}

Matrix<double> Solve(std::string picture_path,
                     std::string sensitivity_path,
                     std::string background_path,
                     std::string solution_path,
                     Parameters<double>& options )
{
#ifdef DEBUG
    std::cerr << "Solve called" << std::endl;
#endif // DEBUG
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "                     Astrophysics Lasso Inverse Abel Solver                     " << std::endl << std::endl;
    std::cout << "Picture:     " << picture_path << std::endl;
    std::cout << "Sensitivity: " << sensitivity_path << std::endl;
    std::cout << "Background:  " << background_path << std::endl;
    std::cout << "Blurring:    " << options.blurring_filter << std::endl;
    std::cout << "Solution:    " << solution_path << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    std::cout << "Running solver with " << omp_get_max_threads() << " threads for parallel computing." << std::endl << std::endl;

    options.model_size = (options.pic_size + 2) * options.pic_size;
    options.MC_max = (size_t) (options.pic_size*options.pic_size / 50.0);
    options.MC_quantile_PF = (size_t) (options.MC_max * (1.0 - 1.0/(std::sqrt(PI*std::log(options.pic_size)))));
    options.MC_quantile_PS = (size_t) (options.MC_max * (1.0 - 1.0/(options.pic_size*options.pic_size)));

    // result matrix containing a solution on each row
    Matrix<double> result_fhat(options.bootstrap_max, options.pic_size);
    Matrix<double> result_fhat_cropped(options.bootstrap_max, std::lround((options.pic_size/2)*(1+1/std::sqrt(2)))-std::lround((options.pic_size/2)*(1-1/std::sqrt(2)))+2);

    // first solution is not bootstrapped
    Matrix<double> picture = CenterOffset(picture_path, 0, 0, options);
    Matrix<double> sensitivity = CenterOffset(sensitivity_path, 0, 0, options);
    Matrix<double> background = CenterOffset(background_path, 0, 0, options);
    background += 1e-4 - background.Min();

    std::random_device rnd;
    std::default_random_engine generator(rnd() + std::chrono::system_clock::now().time_since_epoch().count());

    size_t bootstrap_current = 0;
    while(bootstrap_current < options.bootstrap_max)
    {
        if(bootstrap_current == 0)
        {
            std::cout << std::string(80, '-') << std::endl;
            std::cout << "Computing solution without bootstrapping." << std::endl;
            std::cout << std::string(80, '-') << std::endl << std::endl;
        }
        else
        {
            // bootstrap center of picture
            std::uniform_int_distribution random_center(-100,100);
            int offset_vert = random_center(generator);
            int offset_horiz = random_center(generator);
            picture = CenterOffset(picture_path, offset_vert, offset_horiz, options);
            sensitivity = CenterOffset(sensitivity_path, offset_vert, offset_horiz, options);
            background = CenterOffset(background_path, offset_vert, offset_horiz, options);
            background += 1e-10 - background.Min();

            // resample pixels
            picture = Resample(picture, options.resample_windows_size);

            // choose random wavelets
            std::uniform_int_distribution random_wavelet(0,6);
            options.wavelet[0] = random_wavelet(generator);
            switch(options.wavelet[0])
            {
            case 2:
            {
                std::uniform_int_distribution random_wavelet_param(1,5);
                options.wavelet[1] = random_wavelet_param(generator);
                break;
            }
            case 3:
            {
                std::uniform_int_distribution random_wavelet_param(2,10);
                options.wavelet[1] = 2*random_wavelet_param(generator);
                break;
            }
            case 4:
            {
                std::uniform_int_distribution random_wavelet_param(4,10);
                options.wavelet[1] = random_wavelet_param(generator);
                break;
            }
            case 6:
            {
                std::uniform_int_distribution random_wavelet_param(0,2);
                options.wavelet[1] = 2*random_wavelet_param(generator) + 1;
                break;
            }
            default:
            {
                break;
            }
            }

            // solve with bootstrap
            std::cout << std::string(80, '-') << std::endl;
            std::cout << "Computing with bootstrapping " << "(" << bootstrap_current + 1 << "/" << options.bootstrap_max << "):" << std::endl;
            std::cout << "Wavelets: " << options.wavelet[0] << ", " << options.wavelet[1] << std::endl;
            std::cout << "Center: " << offset_vert << "%, " << offset_horiz << "%" << std::endl;
            std::cout << "Resample windows size: " << options.resample_windows_size << std::endl << std::endl;
            std::cout << std::string(80, '-') << std::endl << std::endl;
        }

        Matrix<double> solution = SolveWS(picture, sensitivity, background, options);
        Matrix<double> fhatw = Matrix<double>(solution.Data(), options.pic_size, options.pic_size, 1);
        Wavelet<double> wave_op = Wavelet<double>((WaveletType)options.wavelet[0], options.wavelet[1]);
        fhatw = wave_op*fhatw;

        Matrix<double> fhats = Matrix<double>(solution.Data()+options.pic_size, options.pic_size, options.pic_size, 1);
        Spline<double> spline_op(options.pic_size);
        fhats = spline_op*fhats;

        Matrix<double> fhat = fhatw + fhats;
        std::copy(fhat.Data(), fhat.Data()+fhat.Length(), result_fhat.Data()+bootstrap_current*fhat.Length());

        Matrix<double> fhat_cropped = fhat.Partial(std::lround((options.pic_size/2)*(1-1/std::sqrt(2)))-1, std::lround((options.pic_size/2)*(1+1/std::sqrt(2))));
        std::copy(fhat_cropped.Data(), fhat_cropped.Data()+fhat_cropped.Length(), result_fhat_cropped.Data()+bootstrap_current*fhat_cropped.Length());

        // write current results to disc
        solution_path << result_fhat_cropped;

        ++bootstrap_current;
    }
#ifdef DEBUG
    std::cerr << "Solve done" << std::endl;
#endif // DEBUG
    return result_fhat_cropped;
}

} // namespace WS
} // namespace alias
