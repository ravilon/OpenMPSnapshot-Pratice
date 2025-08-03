#include "../universal/universal.h"
#include <lodepng.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Part {
short *data;
unsigned short width;
unsigned short height;
unsigned short top_padding;
unsigned short bottom_padding;
unsigned short start_height;
unsigned short end_height;
};

struct Part *divide_image(short *data, unsigned short parts_count,
unsigned short padding, unsigned short width,
unsigned short height, unsigned int thread_num) {
struct Part *out = malloc(sizeof(struct Part));
unsigned short part_height = ceil((double)height / parts_count);
unsigned short part_width = width;
out = malloc(sizeof(struct Part));
out->width = part_width;
out->height = part_height + 2 * padding;
out->bottom_padding = padding;
out->top_padding = padding;
out->start_height = thread_num * part_height;
out->end_height = (thread_num + 1) * part_height;
if (thread_num == 0) {
out->height -= padding;
out->top_padding = 0;
}
if (thread_num == (unsigned int)parts_count - 1) {
out->end_height = height;
out->height = height - out->start_height + padding;
out->bottom_padding = 0;
}

out->data = malloc(out->width * out->height * sizeof(short));
unsigned short start_height = out->start_height - out->top_padding;
for (unsigned short k = start_height;
k < out->end_height + out->bottom_padding; k++) {
for (unsigned short j = 0; j < out->width; j++) {
out->data[(k - start_height) * out->width + j] = data[k * width + j];
}
}
return out;
}

void merge(struct Image *image, struct Part *part) {
for (unsigned short k = part->top_padding;
k < part->height - part->bottom_padding; k++) {
for (unsigned short j = 0; j < part->width; j++) {
short value = part->data[k * part->width + j];
unsigned int index =
((k + part->start_height - part->top_padding) * image->width + j);
image->data[index] = value;
}
}
}

short *conv(struct Part *image, float *kernel, short kernel_size) {
short border = kernel_size / 2;
short *out = malloc(image->width * image->height * sizeof(short));
memset(out, 0, image->width * image->height * sizeof(short));

for (unsigned short i = 0; i < image->width; i++) {
for (unsigned short j = 0; j < image->height; j++) {
unsigned long output_index = j * image->width + i;
for (short m = -border; m <= border; m++) {
for (short n = -border; n <= border; n++) {
if (m + (short)i >= 0 && i + m < image->width && j + (short)n >= 0 &&
j + n < image->height) {
out[output_index] +=
image->data[(j + n) * image->width + i + m] *
kernel[(n + border) * kernel_size + (m + border)];
}
}
}
}
}
return out;
}

float *guassian_kernel(short size, float sigma) {
float *kernel = malloc(size * size * sizeof(float));
short border = size / 2;
float sum = 0;
for (short i = -border; i <= border; i++) {
for (short j = -border; j <= border; j++) {
kernel[(i + border) * size + (j + border)] =
exp(-(i * i + j * j) / (2 * sigma * sigma)) /
(2 * M_PI * sigma * sigma);
sum += kernel[(i + border) * size + (j + border)];
}
}
for (short i = 0; i < size; i++) {
for (short j = 0; j < size; j++) {
kernel[i * size + j] /= sum;
}
}
return kernel;
}

void gaussian_filter(struct Part *image, short kernel_size, float sigma) {
float *kernel = guassian_kernel(kernel_size, sigma);
short *filtered = conv(image, kernel, kernel_size);
free(image->data);
image->data = filtered;
free(kernel);
}

short *sobel_x(struct Part *image) {
float kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
return conv(image, kernel, 3);
}

short *sobel_y(struct Part *image) {
float kernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
return conv(image, kernel, 3);
}

void normalize(short *data, unsigned int size) {
short max = 0, min = 0;
for (unsigned int i = 0; i < size; i++) {
max = max > data[i] ? max : data[i];
min = min < data[i] ? min : data[i];
}
for (unsigned int i = 0; i < size; i++) {
data[i] = (float)(data[i] - min) / (max - min) * 255;
}
}

short *gradinet_direction(short *gradient_x, short *gradient_y,
unsigned short height, short int width) {
short *out = malloc(height * width * sizeof(short));
for (unsigned short i = 0; i < width; i++) {
for (unsigned short j = 0; j < height; j++) {
short gx = gradient_x[j * width + i];
short gy = gradient_y[j * width + i];
out[j * width + i] = atan2(gy, gx) * 180 / M_PI;
}
}
normalize(out, height * width);
return out;
}

short *gradient_intensity(short *gradient_x, short *gradient_y,
short int height, short int width) {
short *out = malloc(height * width * sizeof(short));
for (unsigned short i = 0; i < width; i++) {
for (unsigned short j = 0; j < height; j++) {
short gx = gradient_x[j * width + i];
short gy = gradient_y[j * width + i];
out[j * width + i] = sqrt(gx * gx + gy * gy);
}
}

return out;
}

short *non_maximum(short *gradient_int, short *gradient_dir, short int height,
short int width) {
short *out = malloc(height * width * sizeof(short));
for (unsigned short i = 0; i < width; i++) {
out[i] = 0;
out[(height - 1) * width + i] = 0;
}
for (unsigned short i = 0; i < height; i++) {
out[i * width] = 0;
out[i * width + width - 1] = 0;
}

for (unsigned short i = 1; i < width - 1; i++) {
for (unsigned short j = 1; j < height - 1; j++) {
short dir = gradient_dir[j * width + i];
short value = gradient_int[j * width + i];
short r = 255, q = 255;
if (dir < 0) {
dir += 180;
}
if (dir <= 22 || dir > 157) {
r = gradient_int[(j - 1) * width + i];
q = gradient_int[(j + 1) * width + i];
} else if (dir <= 67) {
r = gradient_int[(j + 1) * width + i - 1];
q = gradient_int[(j - 1) * width + i + 1];
} else if (dir <= 112) {
r = gradient_int[j * width + i - 1];
q = gradient_int[j * width + i + 1];
} else if (dir <= 157) {
r = gradient_int[(j + 1) * width + i + 1];
q = gradient_int[(j - 1) * width + i - 1];
}
if (value >= r && value >= q) {
out[j * width + i] = value;
} else {
out[j * width + i] = 0;
}
}
}
return out;
}

short max(short *data, short int height, short int width) {
short max = 0;
for (unsigned int i = 0; i < (unsigned int)(height * width); i++) {
max = max > data[i] ? max : data[i];
}
return max;
}

short *threshold(short *data, short int height, short int width,
float low_ratio, float high_ratio, short int max) {
short *out = malloc(height * width * sizeof(short));

short high_threshold = max * high_ratio;
short low_threshold = high_threshold * low_ratio;
for (unsigned short i = 0; i < width; i++) {
for (unsigned short j = 0; j < height; j++) {
if (data[j * width + i] >= high_threshold) {
out[j * width + i] = 255;
} else if (data[j * width + i] < low_threshold) {
out[j * width + i] = 0;
} else {
out[j * width + i] = 125;
}
}
}

return out;
}

short *hysterisis(short *data, short int height, short int width) {
short *out = malloc(height * width * sizeof(short));
for (unsigned int i = 0; i < (unsigned int)(height * width); i++) {
out[i] = data[i];
}

for (unsigned short i = 1; i < width - 1; i++) {
for (unsigned short j = 1; j < height - 1; j++) {
if (data[j * width + i] == 125) {
if (data[(j - 1) * width + i - 1] == 255 ||
data[(j - 1) * width + i] == 255 ||
data[(j - 1) * width + i + 1] == 255 ||
data[j * width + i - 1] == 255 || data[j * width + i + 1] == 255 ||
data[(j + 1) * width + i - 1] == 255 ||
data[(j + 1) * width + i] == 255 ||
data[(j + 1) * width + i + 1] == 255) {
out[j * width + i] = 255;
} else {
out[j * width + i] = 0;
}
}
}
}
return out;
}

int main(int argc, char *argv[]) {
short kernel_size = 3;
float low_ratio = 0.2;
float high_ratio = 0.1;
float sigma = 1;
if (argc < 6 && argc != 2) {
printf(
"Usage: %s <filename> <sigma> <kernel_size> <high_ratio> <low_ratio>\n",
argv[0]);
return 1;
}
if (argc == 6) {
sigma = atof(argv[2]);
kernel_size = atoi(argv[3]);
low_ratio = atof(argv[5]);
high_ratio = atof(argv[4]);
}
const char *filename = argv[1];

int therads_num = omp_get_num_procs();
int min_padding = kernel_size / 2 + 1;
short *maxs = malloc(sizeof(short) * therads_num);
short global_max = 0;
struct Image *image = malloc(sizeof(struct Image));
image = decode_image_gray(filename);
double start = omp_get_wtime();
struct Part *part;
#pragma omp parallel firstprivate(therads_num, min_padding, kernel_size,        sigma, low_ratio,                         high_ratio) private(part)                 shared(image, maxs, global_max) num_threads(therads_num)
{
part = divide_image(image->data, therads_num, min_padding, image->width,
image->height, omp_get_thread_num());
#pragma omp barrier
gaussian_filter(part, kernel_size, sigma);
short *gradient_x = sobel_x(part);
short *gradient_y = sobel_y(part);
short *gradient_int =
gradient_intensity(gradient_x, gradient_y, part->height, part->width);
short *gradient_dir =
gradinet_direction(gradient_x, gradient_y, part->height, part->width);
short *non_max =
non_maximum(gradient_int, gradient_dir, part->height, part->width);
short local_max = max(non_max, part->height, part->width);
maxs[omp_get_thread_num()] = local_max;
#pragma omp critical
{
short new_max = 0;
for (int i = 0; i < therads_num; i++) {
new_max = new_max > maxs[i] ? new_max : maxs[i];
}
global_max = new_max;
}
#pragma omp barrier
local_max = global_max;
short *thresholded = threshold(non_max, part->height, part->width,
low_ratio, high_ratio, local_max);
short *hysterisised = hysterisis(thresholded, part->height, part->width);
free(part->data);
part->data = hysterisised;
#pragma omp barrier
merge(image, part);
free(part->data);
free(part);
free(gradient_x);
free(gradient_y);
free(gradient_int);
free(gradient_dir);
free(non_max);
free(thresholded);
}
double end = omp_get_wtime();
printf("Time: %f\n", end - start);
encode_image(image, filename, "_omp");
free(image->data);
free(image);
return 0;
}
