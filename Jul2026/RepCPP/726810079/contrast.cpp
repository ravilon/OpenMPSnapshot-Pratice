#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "hist-equ.h"
#include <mpi.h>
#include <omp.h>

void run_cpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(const char * path);


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    PPM_IMG img_ibuf_c;

    double start_time_total, end_time_total, maxTime;
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Tiempo total de ejecución
    start_time_total = MPI_Wtime();

    //Just checking then max threads and the number of them defined for the execution
    int max_threads = omp_get_max_threads();
    printf("Max number of threads: %d\n", max_threads);
#pragma omp parallel
    {
#pragma omp master
        {
            printf("Number of threads: %d\n", omp_get_num_threads());
        }
    }

    // Contrast enhancement for gray-scale images
    printf("Running contrast enhancement for gray-scale images.\n");
    run_cpu_gray_test("in.pgm");

//    // Contrast enhancement for color images
//    printf("Running contrast enhancement for color images.\n");
//    img_ibuf_c = read_ppm("in.ppm");
//    run_cpu_color_test(img_ibuf_c);
//    free_ppm(img_ibuf_c);

    // Tiempo total de ejecución
    end_time_total = MPI_Wtime();
    double localTime =  end_time_total - start_time_total;

    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Overall processing time: %f (seconds)\n", maxTime * 1000.0/* TIMER */ );
    }

    MPI_Finalize();

    return 0;
}

void run_cpu_color_test(PPM_IMG img_in)
{
//    PPM_IMG img_obuf_hsl, img_obuf_yuv;
//
//    time_t start_test = time(nullptr);
//    printf("Starting CPU processing...\n");
//
//    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
//
//    time_t end_hsl = time(nullptr);
//    double seconds_hsl = difftime(end_hsl, start_test);
//    printf("HSL processing time: %f (sec)\n", seconds_hsl);
//    write_ppm(img_obuf_hsl, "out_hsl.ppm");
//
//    time_t start_yuv = time(nullptr);
//    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
//    time_t end_yuv = time(nullptr);
//    double seconds_yuv = difftime(end_yuv, start_yuv);
//    printf("YUV processing time: %f (sec)\n", seconds_yuv);
//
//    write_ppm(img_obuf_yuv, "out_yuv.ppm");
//
//    free_ppm(img_obuf_hsl);
//    free_ppm(img_obuf_yuv);
//
//    time_t end_test = time(nullptr);
//    double seconds_test = difftime(end_test, start_test);
//    printf("Color test processing time: %f (sec)\n", seconds_test);
}

void run_cpu_gray_test(const char * path)
{
    double maxLocalTimeGray, maxLocalTimeTest;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PGM_IMG_MPI img_in = read_pgm_mpi(path);

    double start_test = MPI_Wtime();
    if (rank == 0) {
        printf("Starting CPU processing...\n");
    }
    PGM_IMG img_obuf = contrast_enhancement_g(img_in);

    double end_gray = MPI_Wtime();
    double localTimeGray = end_gray - start_test;

    MPI_Reduce(&localTimeGray, &maxLocalTimeGray, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Gray processing time: %f (sec)\n", localTimeGray * 1000.0);
    }

    // Only the root process writes the output image
    if (rank == 0) {
        // Assuming the full image has been gathered in the root process
        write_pgm(img_obuf, "out.pgm");
    }

    double end_test = MPI_Wtime();
    double localTimeTest = end_test - start_test;

    MPI_Reduce(&localTimeTest, &maxLocalTimeTest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Gray test processing time: %f (sec)\n", localTimeTest * 1000.0);
    }

    free_pgm(img_obuf);
    free(img_in.segment_in_g);
}

PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];

    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);

    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }

    fclose(in_file);
    free(ibuf);

    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;

    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }

    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG_MPI read_pgm_mpi(const char * path){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PGM_IMG_MPI pgmImgMpiOut;
    PGM_IMG img_ibuf_g;

    int img_size_g,  segment_size_g , remainder_g, width, height;

    if (rank == 0) {
        img_ibuf_g = read_pgm("in.pgm");
        width      = img_ibuf_g.w;
        height     = img_ibuf_g.h;
        img_size_g = img_ibuf_g.w * img_ibuf_g.h;
    }

    MPI_Bcast(&img_size_g, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the size of each segment for grayscale and color images
    segment_size_g = img_size_g / size;
    //segment_size_c = img_size_c / size;

    remainder_g = img_size_g % size;
    //remainder_c = img_size_c % size;

    if (rank == size - 1) {
        segment_size_g += remainder_g; // Last process takes the remainder
    }

    pgmImgMpiOut.height = height;
    pgmImgMpiOut.width = width;
    pgmImgMpiOut.segment_size_g = segment_size_g;

    // Allocate memory for each segment
    pgmImgMpiOut.segment_in_g = (unsigned char *)malloc(segment_size_g * sizeof(unsigned char));

    // Scatter the gray image data
    MPI_Scatter(img_ibuf_g.img, pgmImgMpiOut.segment_size_g, MPI_UNSIGNED_CHAR, pgmImgMpiOut.segment_in_g, pgmImgMpiOut.segment_size_g, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    return pgmImgMpiOut;
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];


    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }

    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);


    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));


    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);
    fclose(in_file);

    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

