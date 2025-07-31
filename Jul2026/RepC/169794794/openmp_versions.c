//
// Created by Hao Xu on 2019-03-07.
//
#include "mosaic.h"

void
process_mosaic_section_openmp(Img * image, Mosaic * mos, int limits[3], Func fff, int pixcel_num, int * total) {
/** limits[0]: the number of mosaic in this mosaic section
limits[1]: the number of pixcel rows in the current mosaic
limits[2]: the number of pixcel columns in the current mosaic*/
int i = 0, rrr = 0, ggg = 0, bbb = 0;

omp_set_num_threads(16);
#pragma omp parallel for private(i) reduction(+: rrr) 
for (i = 0; i < limits[0]; i++)	// the ith cell
{
/****** variables declaraction ******/
unsigned char r, g, b;
int r_sum = 0, g_sum = 0, b_sum = 0;
int index;


for (int j = 0; j < limits[1]; j++)	// the jth row of ith cell
{
// get the index in photo data that strar the jth row of the ith cell
index = fff(i, j, image, mos);

for (int k = 0; k < limits[2]; k++)	// the kth element of jth row
{
r_sum += (int)image->data[index + k * 3];
g_sum += (int)image->data[index + k * 3 + 1];
b_sum += (int)image->data[index + k * 3 + 2];
}
}

// calculate the average
r = (unsigned char)(r_sum / pixcel_num);
g = (unsigned char)(g_sum / pixcel_num);
b = (unsigned char)(b_sum / pixcel_num);

// add to total
rrr += (int)r; 
#pragma omp critical
total[1] += (int)g; 
total[2] += (int)b;	


// mosaic themosaic_vs.exe has triggered a breakpoint. original
for (int j = 0; j < limits[1]; ++j) {      // the jth row of ith cell
// index in data that start the l row in ith cell
index = index = fff(i, j, image, mos);

for (int k = 0; k < limits[2]; ++k) {       // the kth element of jth row
image->data[index + k * 3] = r;
image->data[index + k * 3 + 1] = g;
image->data[index + k * 3 + 2] = b;
}
}

}

total[0] += rrr;

}

void
process_mosaic_section_openmp2(Img * image, Mosaic * mos, int limits[3], Func fff, int pixcel_num, int * total) {

/** limits[0]: the number of mosaic in this mosaic section
limits[1]: the number of pixcel rows in the current mosaic
limits[2]: the number of pixcel columns in the current mosaic*/
int i = 0;
omp_set_num_threads(16);

// construct array of int for r, g, b, each thread can only access an element dependent on its thread number.
int rrr[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
int ggg[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
int bbb[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

//#pragma omp parallel for private(i) schedule(runtime)
//#pragma omp parallel for private(i) schedule(static, 1)
#pragma omp parallel for private(i) schedule(static)
//#pragma omp parallel for private(i) schedule(dynamic)
for (i = 0; i < limits[0]; i++)	// the ith cell
{
/****** variables declaraction ******/
unsigned char r, g, b;
int r_sum = 0, g_sum = 0, b_sum = 0;
int index;


for (int j = 0; j < limits[1]; j++)	// the jth row of ith cell
{
// get the index in photo data that strar the jth row of the ith cell
index = fff(i, j, image, mos);

for (int k = 0; k < limits[2]; k++)	// the kth element of jth row
{
r_sum += (int)image->data[index + k * 3];
g_sum += (int)image->data[index + k * 3 + 1];
b_sum += (int)image->data[index + k * 3 + 2];
}
}

// calculate the average
r = (unsigned char)(r_sum / pixcel_num);
g = (unsigned char)(g_sum / pixcel_num);
b = (unsigned char)(b_sum / pixcel_num);

// add to total
rrr[omp_get_thread_num()] += r;
ggg[omp_get_thread_num()] += g;
bbb[omp_get_thread_num()] += b;

// mosaic the original
for (int j = 0; j < limits[1]; ++j) {      // the jth row of ith cell
// index in data that start the l row in ith cell
index = index = fff(i, j, image, mos);

for (int k = 0; k < limits[2]; ++k) {       // the kth element of jth row
image->data[index + k * 3] = r;
image->data[index + k * 3 + 1] = g;
image->data[index + k * 3 + 2] = b;
}
}

}

for (int m = 0; m < 16; m++)
{
total[0] += rrr[m];
total[1] += ggg[m];
total[2] += bbb[m];
}
};