#include <iostream>
#include <vector>
#include <omp.h>


int compute_julia_value(int width, int height, float x_min, float x_max, float y_min, float y_max, int x, int y)
{
    float real_part, imag_part, constant_real = -0.5, constant_imag = 0.5;
    float temp, real_coord, imag_coord;

    // Map pixel (x, y) to complex plane coordinates
    real_coord = ((float)(width - x - 1) * x_min + (float)(x)*x_max) / (float)(width - 1);
    imag_coord = ((float)(height - y - 1) * y_min + (float)(y)*y_max) / (float)(height - 1);

    // Initialize the complex number (real_part + imag_part * i)
    real_part = real_coord;
    imag_part = imag_coord;

    // Iterate and check if the point escapes (for Julia set)
    for (int iteration = 0; iteration < 1000; iteration++)
    {
        temp = real_part * real_part - imag_part * imag_part + constant_real;
        imag_part = 2 * real_part * imag_part + constant_imag; // Complex multiplication
        real_part = temp;

        // Escape condition: if the value grows beyond a threshold, it won't be part of the set
        if (real_part * real_part + imag_part * imag_part > 1000)
        {
            return 0; // Point escapes, not in the Julia set
        }
    }

    return 1; // Point does not escape, it's in the Julia set
}

void generate_julia_set_parallel(int width, int height, float x_min, float x_max, float y_min, float y_max, std::vector<unsigned char> &rgb)
{
    int julia_value;
    int k;

#pragma omp parallel shared(height, width, x_min, x_max, y_min, y_max, rgb) private(julia_value, k)
    {
#pragma omp for
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                julia_value = compute_julia_value(width, height, x_min, x_max, y_min, y_max, x, y);

                k = 3 * (y * width + x); // Position in RGB buffer
                rgb[k] = 255 * (1 - julia_value);
                rgb[k + 1] = 255 * (1 - julia_value);
                rgb[k + 2] = 255;
            }
        }
    }
}

void generate_julia_set_serial(int width, int height, float x_min, float x_max, float y_min, float y_max, std::vector<unsigned char> &rgb)
{
    int julia_value;
    int k;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            julia_value = compute_julia_value(width, height, x_min, x_max, y_min, y_max, x, y);

            k = 3 * (y * width + x); // Position in RGB buffer
            rgb[k] = 255 * (1 - julia_value);
            rgb[k + 1] = 255 * (1 - julia_value);
            rgb[k + 2] = 255;
        }
    }
}

void write_ppm_image(int width, int height, const std::vector<unsigned char> &rgb, const std::string &filename)
{
    FILE *file_unit = fopen(filename.c_str(), "wb");

    if (!file_unit)
    {
        std::cerr << "Error opening file " << filename << " for writing.\n";
        return;
    }

    // PPM header
    fprintf(file_unit, "P6\n");
    fprintf(file_unit, "%d %d\n", width, height);
    fprintf(file_unit, "255\n");

    // Write the pixel data
    fwrite(rgb.data(), sizeof(unsigned char), 3 * width * height, file_unit);

    fclose(file_unit);

    std::cout << "\nPPM_WRITE:\n";
    std::cout << "  Graphics data saved as '" << filename << "'\n";
}

int main()
{
    int height = 1000;
    int width = 1000;
    double start_time, end_time;
    float x_min = -1.5;
    float x_max = 1.5;
    float y_min = -1.5;
    float y_max = 1.5;

    std::cout << "\nJULIA_SET:\n";
    std::cout << "  C++ version with both serial and OpenMP parallelization.\n";
    std::cout << "  Plot a version of the Julia set for Z(k+1) = Z(k)^2 - 0.8 + 0.156i\n";

    std::vector<unsigned char> rgb(width * height * 3); // RGB buffer

    // Serial Execution
    start_time = omp_get_wtime();
    generate_julia_set_serial(width, height, x_min, x_max, y_min, y_max, rgb);
    end_time = omp_get_wtime() - start_time;
    std::cout << "  Serial execution time: " << end_time << " seconds\n";
    write_ppm_image(width, height, rgb, "julia_serial.ppm");

    // Parallel Execution (OpenMP)
    start_time = omp_get_wtime();
    generate_julia_set_parallel(width, height, x_min, x_max, y_min, y_max, rgb);
    end_time = omp_get_wtime() - start_time;
    std::cout << "  Parallel execution time (OpenMP): " << end_time << " seconds\n";
    write_ppm_image(width, height, rgb, "julia_openmp.ppm");

    // Calculate and display speedup
    double speedup = (end_time > 0) ? (end_time / start_time) : 0;
    std::cout << "  Speedup (Parallel / Serial): " << speedup << "x\n";

    std::cout << "\nJULIA_SET:\n";
    std::cout << "  Normal end of execution.\n";

    return 0;
}