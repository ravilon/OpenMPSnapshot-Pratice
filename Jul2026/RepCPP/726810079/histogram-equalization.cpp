#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

void histogram(int * local_hist, unsigned char * segment_in, int segment_size, int nbr_bin){
    // Initialize local histogram
    memset(local_hist, 0, nbr_bin * sizeof(int));

    // Compute histogram for the received image segment
    #pragma omp parallel for shared(local_hist)
    for (int i = 0; i < segment_size; i++) {
        //Adding this line will produce the same binaries as the sequential ones which is good
        #pragma omp atomic
        local_hist[segment_in[i]]++;
    }
}

void histogram_equalization(unsigned char *segment_out, unsigned char *segment_in,
                            int *global_hist, int segment_size, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;

    #pragma omp for
    for (int i = 0; i < nbr_bin; ++i) {
        if (min == 0) {
            #pragma omp critical
            {
                if (min == 0) {
                    min = global_hist[i];
                }
            }
        }
    }

    d = img_size - min;
    #pragma omp single
    for(int i = 0; i < nbr_bin; i ++){
        cdf += global_hist[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }

    /* Apply the LUT to the segment of the image */
    #pragma omp parallel for shared(lut, segment_in, segment_out)
    for (int i = 0; i < segment_size; i++) {
        int lut_value = lut[segment_in[i]];
        segment_out[i] = lut_value > 255 ? 255 : (unsigned char)lut_value;
    }

    free(lut);
}



