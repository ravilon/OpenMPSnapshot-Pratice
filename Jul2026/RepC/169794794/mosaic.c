#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "mosaic.h"

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acq18hx"        //replace with your user name

//========================================================
//void print_help();
//int process_command_line(int argc, char *argv[]);
//========================================================
typedef enum MODE {
CPU, OPENMP, CUDA, ALL
} MODE;

MODE execution_mode = OPENMP;

unsigned int r_average_sum = 0, g_average_sum = 0, b_average_sum = 0;

int main() {
//========================================================
//	if (process_command_line(argc, argv) == FAILURE)
//		return 1;
//========================================================
// Sample Input Information
unsigned int cell_size = 64;     // 2 ** n
//    execution_mode = CPU;
char input[] = "G:\\github\\GPU_CUDA_openMP\\photo_mosaic\\Binary.ppm";
//    char input[] = "/Users/nyxfer/Documents/GitHub/gpu/photo_mosaic/SheffieldPlainText16x16.ppm";
//    char output[] = "/Users/nyxfer/Documents/GitHub/gpu/photo_mosaic/Sheffield_out.ppm";
PPM write_type = PPM_BINARY;

//	TODO: read input image file (either binary or plain text PPM)

Img *image = read_ppm(input);
//	write_ppm_binary(image, "/Users/nyxfer/Documents/GitHub/gpu/photo_mosaic/hhh.ppm");
execution_mode = CPU;


// -------------------------- pro-process --------------------------------
// cell size
unsigned int cell_num_height = image->height / cell_size;
unsigned int cell_remain_height = image->height % cell_size;

unsigned int cell_num_weight = image->width / cell_size;
unsigned int cell_remain_weight = image->width % cell_size;

// temp r g b for each cell
unsigned int num_main_call = cell_num_height * cell_num_weight;
unsigned int num_in_cell = cell_size * cell_size;
unsigned char r, g, b;

// set pointer
unsigned int index = 0;     // the first index for each element start a row in cell
unsigned int r_sum = 0, g_sum = 0, b_sum = 0;


// --------------------------------------------
//    unsigned char d[5 * 7 * 3];
//    for (int i = 0; i < 5; i++) {
//        for (int j = 0; j < 7; ++j) {
//            d[(i * 7 + j) * 3] = (unsigned char) j; // where "val" is some value.
//            d[(i * 7 + j) * 3 + 1] = (unsigned char) j;
//            d[(i * 7 + j) * 3 + 2] = (unsigned char) j;
//        }
//    }
//
//    for (int i = 0; i < 5; i++) {
//        for (int j = 0; j < 7; ++j) {
//            printf("%hhu, %hhu, %hhu\t\t", d[(i * 7 + j) * 3], d[(i * 7 + j) * 3 + 1], d[(i * 7 + j) * 3 + 2]);
//        }
//        printf("\n");
//    }
//
//    Img im;
//    im.width = 7;
//    im.height = 5;
//    im.num_pixel = 35;
//    im.data = d;
//    Img *image = &im;
// --------------------------------------------
// data
unsigned char *p_data = image->data;
// print data
//    for (int i = 0; i < image->height; i++) {
//        for (int j = 0; j < image->width; ++j) {
//            printf("%hhu, %hhu, %hhu\t\t", p_data[(i * 7 + j) * 3], p_data[(i * 7 + j) * 3 + 1], p_data[(i * 7 + j) * 3 + 2]);
//        }
//        printf("\n");
//    }

execution_mode = CPU;
//	//TODO: execute the mosaic filter based on the mode
switch (execution_mode) {
case CPU: {
//starting timing here
clock_t timer = clock();

// process the main section
for (unsigned int i = 0; i < num_main_call; ++i) {      // the ith cell
// initial the sum of r, g, b
r_sum = 0;
g_sum = 0;
b_sum = 0;

for (unsigned int j = 0; j < cell_size; ++j) {      // the jth row of ith cell
// index in data that start the l row in ith cell
index = ((i / cell_num_weight * cell_size + j) * image->width + (i % cell_num_weight) * cell_size) *
3;

for (unsigned int k = 0; k < cell_size; ++k) {           // the kth element of jth row
r_sum += p_data[index + k * 3];
g_sum += p_data[index + k * 3 + 1];
b_sum += p_data[index + k * 3 + 2];
}
}

// calculate the average
r = (unsigned char)(r_sum / num_in_cell);
g = (unsigned char)(g_sum / num_in_cell);
b = (unsigned char)(b_sum / num_in_cell);

// update average sum
r_average_sum += r;
g_average_sum += g;
b_average_sum += b;

//        printf("\n%hhu, %hhu, %hhu\n", r, g, b);

// mosaic the original
for (unsigned int j = 0; j < cell_size; ++j) {      // the jth row of ith cell
// index in data that start the l row in ith cell
index = ((i / cell_num_weight * cell_size + j) * image->width + (i % cell_num_weight) * cell_size) *
3;

for (unsigned int k = 0; k < cell_size; ++k) {           // the kth element of jth row
p_data[index + k * 3] = r;
p_data[index + k * 3 + 1] = g;
p_data[index + k * 3 + 2] = b;
}
}
}

// edge case: column edge
if (cell_remain_weight != 0) {
for (unsigned int m = 0; m < cell_num_height; ++m) {
r_sum = 0;
g_sum = 0;
b_sum = 0;
for (unsigned int n = 0; n < cell_size; ++n) {
index = (m * image->width * cell_size + n * image->width + cell_num_weight * cell_size) * 3;
for (unsigned int l = 0; l < cell_remain_weight; ++l) {
r_sum += p_data[index + l * 3];
g_sum += p_data[index + l * 3 + 1];
b_sum += p_data[index + l * 3 + 2];
}
}
// calculate the average
num_in_cell = cell_remain_weight * cell_size;
r = (unsigned char)(r_sum / num_in_cell);
g = (unsigned char)(g_sum / num_in_cell);
b = (unsigned char)(b_sum / num_in_cell);

for (unsigned int n = 0; n < cell_size; ++n) {
index = (m * image->width * cell_size + n * image->width + cell_num_weight * cell_size) * 3;
for (unsigned int q = 0; q < cell_remain_weight; ++q) {
p_data[index + q * 3] = r;
p_data[index + q * 3 + 1] = g;
p_data[index + q * 3 + 2] = b;
}
}
}
}

//     edge case: row edge
if (cell_remain_height != 0) {
for (unsigned int m = 0; m < cell_num_weight; ++m) {
r_sum = 0;
g_sum = 0;
b_sum = 0;
for (unsigned int p = 0; p < cell_remain_height; ++p) {
index = (image->width * (cell_num_height * cell_size + p) + m * cell_size) * 3;
for (unsigned int n = 0; n < cell_size; ++n) {
r_sum += p_data[index + n * 3];
g_sum += p_data[index + n * 3 + 1];
b_sum += p_data[index + n * 3 + 2];
}
}

// calculate the average
num_in_cell = cell_size * cell_remain_height;
r = (unsigned char)(r_sum / num_in_cell);
g = (unsigned char)(g_sum / num_in_cell);
b = (unsigned char)(b_sum / num_in_cell);

for (unsigned int p = 0; p < cell_remain_height; ++p) {
index = (image->width * (cell_num_height * cell_size + p) + m * cell_size) * 3;
for (unsigned int n = 0; n < cell_size; ++n) {
p_data[index + n * 3] = r;
p_data[index + n * 3 + 1] = g;
p_data[index + n * 3 + 2] = b;
}
}
}


r_sum = 0;
g_sum = 0;
b_sum = 0;
// spacial case of last mosaic cell
for (unsigned int i = 0; i < cell_remain_height; ++i) {
index = ((cell_num_height * cell_size + i) * image->width + cell_num_weight * cell_size) * 3;
for (unsigned int j = 0; j < cell_remain_weight; ++j) {
r_sum += p_data[index + j * 3];
g_sum += p_data[index + j * 3 + 1];
b_sum += p_data[index + j * 3 + 2];
}
}
// calculate the average
num_in_cell = cell_remain_height * cell_remain_weight;
r = (unsigned char)(r_sum / num_in_cell);
g = (unsigned char)(g_sum / num_in_cell);
b = (unsigned char)(b_sum / num_in_cell);

for (unsigned int i = 0; i < cell_remain_height; ++i) {
index = ((cell_num_height * cell_size + i) * image->width + cell_num_weight * cell_size) * 3;
for (unsigned int j = 0; j < cell_remain_weight; ++j) {
p_data[index + j * 3] = r;
p_data[index + j * 3 + 1] = g;
p_data[index + j * 3 + 2] = b;
}
}
}

//    printf("\n");
//    for (int l = 0; l < 35*3; ++l) {
//        printf("%hhu  ", p_data[l]);
//    }

//    for (int i = 0; i < image->height; i++) {
//        for (int j = 0; j < image->width; ++j) {
//            printf("%hhu, %hhu, %hhu\t\t", p_data[(i * 7 + j) * 3], p_data[(i * 7 + j) * 3 + 1], p_data[(i * 7 + j) * 3 + 2]);
//        }
//        printf("\n");
//    }
//
//    printf("\n %d/t/t%d ", image->height, image->width);
//
//			// TODO: Output the average colour value for the image
int num_average = cell_num_height * cell_num_weight;
if (cell_remain_weight != 0) num_average += cell_num_height;
if (cell_remain_height != 0) num_average += cell_num_weight + 1;
printf("CPU Average image colour red = %d, green = %d, blue = %d \n",
r_average_sum / num_average, g_average_sum / num_average, b_average_sum / num_average);
//
//			//end timing here
clock_t timer_end = clock();
printf("constant CLOCKS_PER_SEC is: %lf", (double)(timer_end - timer));
double cost = (double)(timer_end - timer) / CLOCKS_PER_SEC;
printf("constant CLOCKS_PER_SEC is: %d, time cost is: %lf secs", CLOCKS_PER_SEC, cost);

//            printf("CPU mode execution time took ??? s and ???ms\n");
break;
}
case OPENMP: {
//TODO: starting timing here
double time_begin = omp_get_wtime();
// #pragma omp parallel for schedule(static, 1) private(r_sum, g_sum, b_sum) reduction(+:r_average_sum, +:g_average_sum, +:b_average_sum)


//TODO: calculate the average colour value
// process the main section


//			// Output the average colour value for the image
int num_average = cell_num_height * cell_num_weight;
if (cell_remain_weight != 0) num_average += cell_num_height;
if (cell_remain_height != 0) num_average += cell_num_weight + 1;
printf("CPU Average image colour red = %d, green = %d, blue = %d \n",
r_average_sum / num_average, g_average_sum / num_average, b_average_sum / num_average);

//TODO: end timing here
double time_end = omp_get_wtime();
printf("constant CLOCKS_PER_SEC is: %lf\n", (double)(time_end - time_begin));
double cost = (double)(time_end - time_begin) / CLOCKS_PER_SEC;
printf("constant CLOCKS_PER_SEC is: %d, time cost is: %lf secs", CLOCKS_PER_SEC, cost);
//            printf("OPENMP mode execution time took ??? s and ?? ?ms\n");
break;
}

case (CUDA): {
printf("CUDA Implementation not required for assignment part 1\n");
break;
}

case (ALL): {
//TODO
break;
}
}


//save the output image file (from last executed mode)
image->data = p_data;
write_ppm_binary(image, "G:\\github\\GPU_CUDA_openMP\\photo_mosaic\\hhhhh\\h.ppm");

return 0;
}
//========================================================
//void print_help(){
//	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);
//
//	printf("where:\n");
//	printf("\tC              Is the mosaic cell size which should be any positive\n"
//		   "\t               power of 2 number \n");
//	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
//		   "\t               ALL. The mode specifies which version of the simulation\n"
//		   "\t               code should execute. ALL should execute each mode in\n"
//		   "\t               turn.\n");
//	printf("\t-i input_file  Specifies an input image file\n");
//	printf("\t-o output_file Specifies an output image file which will be used\n"
//		   "\t               to write the mosaic image\n");
//	printf("[options]:\n");
//	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
//		   "\t               PPM_PLAIN_TEXT\n ");
//}
//
//int process_command_line(int argc, char *argv[]){
//	if (argc < 7){
//		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
//		print_help();
//		return FAILURE;
//	}
//
//	//first argument is always the executable name
//
//	//read in the non optional command line arguments
//	c = (unsigned int)atoi(argv[1]);
//
//	//TODO: read in the mode
//
//	//TODO: read in the input image name
//
//	//TODO: read in the output image name
//
//	//TODO: read in any optional part 3 arguments
//
//	return SUCCESS;
//}#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "mosaic.h"

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acq18hx"        //replace with your user name

//========================================================
//void print_help();
//int process_command_line(int argc, char *argv[]);
//========================================================

typedef enum MODE {
CPU, OPENMP, CUDA, ALL
} MODE;

MODE execution_mode;


int main() {
//========================================================
//	if (process_command_line(argc, argv) == FAILURE)
//		return 1;
//========================================================
// Sample Input Information
unsigned int cell_size = 64;     // 2 ** n
//    execution_mode = CPU;
char input[] = "G:\\github\\GPU_CUDA_openMP\\photo_mosaic\\Binary.ppm";
//    char input[] = "/Users/nyxfer/Documents/GitHub/gpu/photo_mosaic/SheffieldPlainText16x16.ppm";
//    char output[] = "/Users/nyxfer/Documents/GitHub/gpu/photo_mosaic/Sheffield_out.ppm";
PPM write_type = PPM_BINARY;

//	TODO: read input image file (either binary or plain text PPM)

Img *image = read_ppm(input);
//	write_ppm_binary(image, "/Users/nyxfer/Documents/GitHub/gpu/photo_mosaic/hhh.ppm");
execution_mode = OPENMP;


// -------------------------- pro-process --------------------------------
// cell size
unsigned int cell_num_height = image->height / cell_size;
unsigned int cell_remain_height = image->height % cell_size;

unsigned int cell_num_weight = image->width / cell_size;
unsigned int cell_remain_weight = image->width % cell_size;

// temp r g b for each cell
unsigned int num_main_call = cell_num_height * cell_num_weight;
unsigned int num_in_cell = cell_size * cell_size;

// --------------------------------------------
//    unsigned char d[5 * 7 * 3];
//    for (int i = 0; i < 5; i++) {
//        for (int j = 0; j < 7; ++j) {
//            d[(i * 7 + j) * 3] = (unsigned char) j; // where "val" is some value.
//            d[(i * 7 + j) * 3 + 1] = (unsigned char) j;
//            d[(i * 7 + j) * 3 + 2] = (unsigned char) j;
//        }
//    }
//
//    for (int i = 0; i < 5; i++) {
//        for (int j = 0; j < 7; ++j) {
//            printf("%hhu, %hhu, %hhu\t\t", d[(i * 7 + j) * 3], d[(i * 7 + j) * 3 + 1], d[(i * 7 + j) * 3 + 2]);
//        }
//        printf("\n");
//    }
//
//    Img im;
//    im.width = 7;
//    im.height = 5;
//    im.num_pixel = 35;
//    im.data = d;
//    Img *image = &im;
// --------------------------------------------
// data

// set pointer
unsigned int index = 0;     // the first index for each element start a row in cell
unsigned char r, g, b;
unsigned int r_average_sum = 0, g_average_sum = 0, b_average_sum = 0;
unsigned int r_sum = 0, g_sum = 0, b_sum = 0;

unsigned char *p_data = image->data;
// print data
//    for (int i = 0; i < image->height; i++) {
//        for (int j = 0; j < image->width; ++j) {
//            printf("%hhu, %hhu, %hhu\t\t", p_data[(i * 7 + j) * 3], p_data[(i * 7 + j) * 3 + 1], p_data[(i * 7 + j) * 3 + 2]);
//        }
//        printf("\n");
//    }

execution_mode = CPU;
//	//TODO: execute the mosaic filter based on the mode
switch (execution_mode) {
case CPU: {


//starting timing here
clock_t timer = clock();

// process the main section
for (unsigned int i = 0; i < num_main_call; ++i) {      // the ith cell
// initial the sum of r, g, b
r_sum = 0;
g_sum = 0;
b_sum = 0;

for (unsigned int j = 0; j < cell_size; ++j) {      // the jth row of ith cell
// index in data that start the l row in ith cell
index = ((i / cell_num_weight * cell_size + j) * image->width + (i % cell_num_weight) * cell_size) *
3;

for (unsigned int k = 0; k < cell_size; ++k) {           // the kth element of jth row
r_sum += p_data[index + k * 3];
g_sum += p_data[index + k * 3 + 1];
b_sum += p_data[index + k * 3 + 2];
}
}

// calculate the average
r = (unsigned char)(r_sum / num_in_cell);
g = (unsigned char)(g_sum / num_in_cell);
b = (unsigned char)(b_sum / num_in_cell);

// update average sum
r_average_sum += r;
g_average_sum += g;
b_average_sum += b;

//        printf("\n%hhu, %hhu, %hhu\n", r, g, b);

// mosaic the original
for (unsigned int j = 0; j < cell_size; ++j) {      // the jth row of ith cell
// index in data that start the l row in ith cell
index = ((i / cell_num_weight * cell_size + j) * image->width + (i % cell_num_weight) * cell_size) *
3;

for (unsigned int k = 0; k < cell_size; ++k) {           // the kth element of jth row
p_data[index + k * 3] = r;
p_data[index + k * 3 + 1] = g;
p_data[index + k * 3 + 2] = b;
}
}
}

// edge case: column edge
if (cell_remain_weight != 0) {
for (unsigned int m = 0; m < cell_num_height; ++m) {
r_sum = 0;
g_sum = 0;
b_sum = 0;
for (unsigned int n = 0; n < cell_size; ++n) {
index = (m * image->width * cell_size + n * image->width + cell_num_weight * cell_size) * 3;
for (unsigned int l = 0; l < cell_remain_weight; ++l) {
r_sum += p_data[index + l * 3];
g_sum += p_data[index + l * 3 + 1];
b_sum += p_data[index + l * 3 + 2];
}
}
// calculate the average
num_in_cell = cell_remain_weight * cell_size;
r = (unsigned char)(r_sum / num_in_cell);
g = (unsigned char)(g_sum / num_in_cell);
b = (unsigned char)(b_sum / num_in_cell);

for (unsigned int n = 0; n < cell_size; ++n) {
index = (m * image->width * cell_size + n * image->width + cell_num_weight * cell_size) * 3;
for (unsigned int q = 0; q < cell_remain_weight; ++q) {
p_data[index + q * 3] = r;
p_data[index + q * 3 + 1] = g;
p_data[index + q * 3 + 2] = b;
}
}
}
}

//     edge case: row edge
if (cell_remain_height != 0) {
for (unsigned int m = 0; m < cell_num_weight; ++m) {
r_sum = 0;
g_sum = 0;
b_sum = 0;
for (unsigned int p = 0; p < cell_remain_height; ++p) {
index = (image->width * (cell_num_height * cell_size + p) + m * cell_size) * 3;
for (unsigned int n = 0; n < cell_size; ++n) {
r_sum += p_data[index + n * 3];
g_sum += p_data[index + n * 3 + 1];
b_sum += p_data[index + n * 3 + 2];
}
}

// calculate the average
num_in_cell = cell_size * cell_remain_height;
r = (unsigned char)(r_sum / num_in_cell);
g = (unsigned char)(g_sum / num_in_cell);
b = (unsigned char)(b_sum / num_in_cell);

for (unsigned int p = 0; p < cell_remain_height; ++p) {
index = (image->width * (cell_num_height * cell_size + p) + m * cell_size) * 3;
for (unsigned int n = 0; n < cell_size; ++n) {
p_data[index + n * 3] = r;
p_data[index + n * 3 + 1] = g;
p_data[index + n * 3 + 2] = b;
}
}
}


r_sum = 0;
g_sum = 0;
b_sum = 0;
// spacial case of last mosaic cell
for (unsigned int i = 0; i < cell_remain_height; ++i) {
index = ((cell_num_height * cell_size + i) * image->width + cell_num_weight * cell_size) * 3;
for (unsigned int j = 0; j < cell_remain_weight; ++j) {
r_sum += p_data[index + j * 3];
g_sum += p_data[index + j * 3 + 1];
b_sum += p_data[index + j * 3 + 2];
}
}
// calculate the average
num_in_cell = cell_remain_height * cell_remain_weight;
r = (unsigned char)(r_sum / num_in_cell);
g = (unsigned char)(g_sum / num_in_cell);
b = (unsigned char)(b_sum / num_in_cell);

for (unsigned int i = 0; i < cell_remain_height; ++i) {
index = ((cell_num_height * cell_size + i) * image->width + cell_num_weight * cell_size) * 3;
for (unsigned int j = 0; j < cell_remain_weight; ++j) {
p_data[index + j * 3] = r;
p_data[index + j * 3 + 1] = g;
p_data[index + j * 3 + 2] = b;
}
}
}

//    printf("\n");
//    for (int l = 0; l < 35*3; ++l) {
//        printf("%hhu  ", p_data[l]);
//    }

//    for (int i = 0; i < image->height; i++) {
//        for (int j = 0; j < image->width; ++j) {
//            printf("%hhu, %hhu, %hhu\t\t", p_data[(i * 7 + j) * 3], p_data[(i * 7 + j) * 3 + 1], p_data[(i * 7 + j) * 3 + 2]);
//        }
//        printf("\n");
//    }
//
//    printf("\n %d/t/t%d ", image->height, image->width);
//
//			// TODO: Output the average colour value for the image
int num_average = cell_num_height * cell_num_weight;
if (cell_remain_weight != 0) num_average += cell_num_height;
if (cell_remain_height != 0) num_average += cell_num_weight + 1;
printf("CPU Average image colour red = %d, green = %d, blue = %d \n",
r_average_sum / num_average, g_average_sum / num_average, b_average_sum / num_average);
//
//			//end timing here
clock_t timer_end = clock();
printf("constant CLOCKS_PER_SEC is: %lf", (double)(timer_end - timer));
double cost = (double)(timer_end - timer) / CLOCKS_PER_SEC;
printf("constant CLOCKS_PER_SEC is: %d, time cost is: %lf secs", CLOCKS_PER_SEC, cost);

//            printf("CPU mode execution time took ??? s and ???ms\n");
break;
}
case OPENMP: {

int max_threads = omp_get_max_threads();
printf("OpenMP using %d threads\n", max_threads);


//TODO: starting timing here
double time_begin = omp_get_wtime();
// #pragma omp parallel for schedue(static, 1) private(r_sum, g_sum, b_sum) reduction(+:r_average_sum, +:g_average_sum, +:b_average_sum)


//TODO: calculate the average colour value
// process the main section


//			// Output the average colour value for the image
int num_average = cell_num_height * cell_num_weight;
if (cell_remain_weight != 0) num_average += cell_num_height;
if (cell_remain_height != 0) num_average += cell_num_weight + 1;
printf("CPU Average image colour red = %d, green = %d, blue = %d \n",
r_average_sum / num_average, g_average_sum / num_average, b_average_sum / num_average);

//TODO: end timing here
double time_end = omp_get_wtime();
printf("constant CLOCKS_PER_SEC is: %lf\n", (double)(time_end - time_begin));
double cost = (double)(time_end - time_begin) / CLOCKS_PER_SEC;
printf("constant CLOCKS_PER_SEC is: %d, time cost is: %lf secs", CLOCKS_PER_SEC, cost);
//            printf("OPENMP mode execution time took ??? s and ?? ?ms\n");
break;
}

case (CUDA): {
printf("CUDA Implementation not required for assignment part 1\n");
break;
}

case (ALL): {
//TODO
break;
}
}


//save the output image file (from last executed mode)
image->data = p_data;
write_ppm_binary(image, "G:\\github\\GPU_CUDA_openMP\\photo_mosaic\\hhhhh\\h.ppm");

return 0;
}
//========================================================
//void print_help(){
//	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);
//
//	printf("where:\n");
//	printf("\tC              Is the mosaic cell size which should be any positive\n"
//		   "\t               power of 2 number \n");
//	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
//		   "\t               ALL. The mode specifies which version of the simulation\n"
//		   "\t               code should execute. ALL should execute each mode in\n"
//		   "\t               turn.\n");
//	printf("\t-i input_file  Specifies an input image file\n");
//	printf("\t-o output_file Specifies an output image file which will be used\n"
//		   "\t               to write the mosaic image\n");
//	printf("[options]:\n");
//	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
//		   "\t               PPM_PLAIN_TEXT\n ");
//}
//
//int process_command_line(int argc, char *argv[]){
//	if (argc < 7){
//		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
//		print_help();
//		return FAILURE;
//	}
//
//	//first argument is always the executable name
//
//	//read in the non optional command line arguments
//	c = (unsigned int)atoi(argv[1]);
//
//	//TODO: read in the mode
//
//	//TODO: read in the input image name
//
//	//TODO: read in the output image name
//
//	//TODO: read in any optional part 3 arguments
//
//	return SUCCESS;
//}