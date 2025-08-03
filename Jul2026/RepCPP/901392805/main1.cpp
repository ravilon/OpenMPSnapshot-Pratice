#include <iostream>
#include <vector>
#include <complex>
#include <omp.h>
#include <string>

using namespace std;

const int MAX_ITERATIONS = 1000;

// Color mapping function
void apply_color(int iteration, int max_iteration, unsigned char &r, unsigned char &g, unsigned char &b)
{
if (iteration == max_iteration)
{
r = g = b = 255;
}
else
{
float t = (float)iteration / max_iteration;
r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}
}

// Mandelbrot computation for a single point
int mandelbrot(int width, int height, float x_min, float x_max, float y_min, float y_max, int x, int y)
{
float real = ((float)x / width) * (x_max - x_min) + x_min;
float imag = ((float)y / height) * (y_max - y_min) + y_min;

complex<float> point(real, imag);
//complex<float> point(0, 16);

complex<float> z(0, 0);

int iteration = 0;
while (abs(z) <= 2 && iteration < MAX_ITERATIONS)
{
z = z * z + point;
iteration++;
}
return iteration;
}

// Serial Mandelbrot generation
void generate_mandelbrot_serial(int width, int height, float x_min, float x_max, float y_min, float y_max, vector<unsigned char> &rgb)
{
for (int y = 0; y < height; y++)
{
for (int x = 0; x < width; x++)
{
int mandelbrot_value = mandelbrot(width, height, x_min, x_max, y_min, y_max, x, y);

unsigned char r, g, b;
apply_color(mandelbrot_value, MAX_ITERATIONS, r, g, b);

int k = 3 * (y * width + x);
rgb[k] = r;
rgb[k + 1] = g;
rgb[k + 2] = b;
}
}
}

// Parallel Mandelbrot generation using OpenMP
void generate_mandelbrot_parallel(int width, int height, float x_min, float x_max, float y_min, float y_max, vector<unsigned char> &rgb)
{
#pragma omp parallel for schedule(dynamic)
for (int y = 0; y < height; y++)
{
for (int x = 0; x < width; x++)
{
int mandelbrot_value = mandelbrot(width, height, x_min, x_max, y_min, y_max, x, y);

unsigned char r, g, b;
apply_color(mandelbrot_value, MAX_ITERATIONS, r, g, b);

int k = 3 * (y * width + x);
rgb[k] = r;
rgb[k + 1] = g;
rgb[k + 2] = b;
}
}
}

// Write RGB data to a PPM file
void write_ppm_image(int width, int height, const vector<unsigned char> &rgb, const string &filename)
{
FILE *file_unit = fopen(filename.c_str(), "wb");

if (!file_unit)
{
cerr << "Error opening file " << filename << " for writing.\n";
return;
}

fprintf(file_unit, "P6\n%d %d\n255\n", width, height);
fwrite(rgb.data(), sizeof(unsigned char), 3 * width * height, file_unit);
fclose(file_unit);
}

// int main()
// {
//     int width = 1000, height = 1000;
//     float x_min = -2.0, x_max = 1.0, y_min = -1.5, y_max = 1.5;

//     vector<unsigned char> rgb(width * height * 3);

//     // Serial execution
//     double start_time = omp_get_wtime();
//     generate_mandelbrot_serial(width, height, x_min, x_max, y_min, y_max, rgb);
//     double end_time = omp_get_wtime();
//     double serial_time = end_time - start_time;
//     cout << "Serial execution time: " << (serial_time) << " seconds\n";
//     write_ppm_image(width, height, rgb, "mandelbrot_serial.ppm");

//     // Parallel execution
//     start_time = omp_get_wtime();
//     generate_mandelbrot_parallel(width, height, x_min, x_max, y_min, y_max, rgb);
//     end_time = omp_get_wtime();
//     double parallel_time = end_time - start_time;
//     cout << "Parallel execution time (OpenMP): " << (parallel_time) << " seconds\n";
//     write_ppm_image(width, height, rgb, "mandelbrot_openmp.ppm");

//     double speedup = serial_time / parallel_time;

//     cout << "Speedup (Serial / Parallel): " << speedup << endl;

//     return 0;
// }
int main()
{
int width = 1000, height = 1000;
float x_min = -2.0, x_max = 1.0, y_min = -1.5, y_max = 1.5;

vector<unsigned char> rgb(width * height * 3);

// Zoom parameters
float zoom_factor =0.8; // f < 1 = zoom in | f > 1 = zoom out
float center_x = -0.75;  // Center of zoom (real part)
float center_y = 0.0;    // Center of zoom (imaginary part)
int zoom_iterations = 3;

for (int i = 0; i < zoom_iterations; ++i)
{
cout << "Zoom iteration " << i + 1 << "...\n";

// Adjust bounds for zoom
float x_range = (x_max - x_min) * zoom_factor;
float y_range = (y_max - y_min) * zoom_factor;
x_min = center_x - x_range / 2;
x_max = center_x + x_range / 2;
y_min = center_y - y_range / 2;
y_max = center_y + y_range / 2;

// Serial execution
double start_time = omp_get_wtime();
generate_mandelbrot_serial(width, height, x_min, x_max, y_min, y_max, rgb);
double end_time = omp_get_wtime();
double serial_time = end_time - start_time;
cout << "Serial execution time: " << serial_time << " seconds\n";

string serial_filename = "mandelbrot_serial_zoom_" + to_string(i + 1) + ".ppm";
write_ppm_image(width, height, rgb, serial_filename);
cout << "Saved serial image: " << serial_filename << "\n";

// Parallel execution
start_time = omp_get_wtime();
generate_mandelbrot_parallel(width, height, x_min, x_max, y_min, y_max, rgb);
end_time = omp_get_wtime();
double parallel_time = end_time - start_time;
cout << "Parallel execution time: " << parallel_time << " seconds\n";

string parallel_filename = "mandelbrot_parallel_zoom_" + to_string(i + 1) + ".ppm";
write_ppm_image(width, height, rgb, parallel_filename);
cout << "Saved parallel image: " << parallel_filename << "\n";

// Compute speedup
double speedup = serial_time / parallel_time;
cout << "Speedup (Serial / Parallel): " << speedup << "\n";
}

return 0;
}
