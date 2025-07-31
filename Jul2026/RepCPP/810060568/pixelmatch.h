#ifndef PIXELMATCH_HPP
#define PIXELMATCH_HPP

#include <stdint.h>

#ifdef _OPENMP
  // OpenMP is enabled
  #define OPENMP_ENABLED 1
#else
  // OpenMP is not enabled
  #define OPENMP_ENABLED 0
#endif

#ifndef MIN
  #define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
  #define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

// pixelmatch_blend semi-transparent color with white
uint8_t pixelmatch_blend(uint8_t c, double a) {
  return 255 + (c - 255) * a;
}

double pixelmatch_rgb_2y(uint8_t r, uint8_t g, uint8_t b) { return r * 0.29889531 + g * 0.58662247 + b * 0.11448223; }
double pixelmatch_rgb_2i(uint8_t r, uint8_t g, uint8_t b) { return r * 0.59597799 - g * 0.27417610 - b * 0.32180189; }
double pixelmatch_rgb_2q(uint8_t r, uint8_t g, uint8_t b) { return r * 0.21147017 - g * 0.52261711 + b * 0.31114694; }

// calculate color difference according to the paper "Measuring perceived color difference
// using YIQ NTSC transmission color space in mobile applications" by Y. Kotsarenko and F. Ramos

double color_delta(const uint8_t* img1, const uint8_t* img2, size_t k, size_t m, int yOnly) {
  double a1 = double(img1[k + 3]) / 255;
  double a2 = double(img2[m + 3]) / 255;

  uint8_t r1 = pixelmatch_blend(img1[k + 0], a1);
  uint8_t g1 = pixelmatch_blend(img1[k + 1], a1);
  uint8_t b1 = pixelmatch_blend(img1[k + 2], a1);

  uint8_t r2 = pixelmatch_blend(img2[m + 0], a2);
  uint8_t g2 = pixelmatch_blend(img2[m + 1], a2);
  uint8_t b2 = pixelmatch_blend(img2[m + 2], a2);

  double y = pixelmatch_rgb_2y(r1, g1, b1) - pixelmatch_rgb_2y(r2, g2, b2);

  if (yOnly) return y; // brightness difference only

  double i = pixelmatch_rgb_2i(r1, g1, b1) - pixelmatch_rgb_2i(r2, g2, b2);
  double q = pixelmatch_rgb_2q(r1, g1, b1) - pixelmatch_rgb_2q(r2, g2, b2);

  return 0.5053 * y * y + 0.299 * i * i + 0.1957 * q * q;
}

// check if a pixel is likely a part of anti-aliasing;
// based on "Anti-aliased Pixel and Intensity Slope Detector" paper by V. Vysniauskas, 2009

int pixelmatch_is_antialiased(const uint8_t* img, size_t x1, size_t y1, size_t width, size_t height, const uint8_t* img2) {
  size_t x0          = x1 > 0 ? x1 - 1 : 0;
  size_t y0          = y1 > 0 ? y1 - 1 : 0;
  size_t x2          = MIN(x1 + 1, width - 1);
  size_t y2          = MIN(y1 + 1, height - 1);
  size_t pos         = (y1 * width + x1) * 4;
  uint64_t zeroes    = 0;
  uint64_t positives = 0;
  uint64_t negatives = 0;
  double min         = 0;
  double max         = 0;
  size_t minX = 0, minY = 0, maxX = 0, maxY = 0;

  // go through 8 adjacent pixels
  for (size_t x = x0; x <= x2; x++) {
    for (size_t y = y0; y <= y2; y++) {
      if (x == x1 && y == y1) continue;

      // brightness delta between the center pixel and adjacent one
      double delta = color_delta(img, img, pos, (y * width + x) * 4, true);

      // count the number of equal, darker and brighter adjacent pixels
      if (delta == 0)
        zeroes++;
      else if (delta < 0)
        negatives++;
      else if (delta > 0)
        positives++;

      // if found more than 2 equal siblings, it's definitely not anti-aliasing
      if (zeroes > 2) return 0;

      if (!img2) continue;

      // remember the darkest pixel
      if (delta < min) {
        min  = delta;
        minX = x;
        minY = y;
      }
      // remember the brightest pixel
      if (delta > max) {
        max  = delta;
        maxX = x;
        maxY = y;
      }
    }
  }

  if (!img2) return true;

  // if there are no both darker and brighter pixels among siblings, it's not anti-aliasing
  if (negatives == 0 || positives == 0) return 0;

  // if either the darkest or the brightest pixel has more than 2 equal siblings in both images
  // (definitely not anti-aliased), this pixel is anti-aliased
  return (!pixelmatch_is_antialiased(img, minX, minY, width, height, NULL) && !pixelmatch_is_antialiased(img2, minX, minY, width, height, NULL)) ||
         (!pixelmatch_is_antialiased(img, maxX, maxY, width, height, NULL) && !pixelmatch_is_antialiased(img2, maxX, maxY, width, height, NULL));
}

#if OPENMP_ENABLED
uint64_t pixelmatch(const uint8_t* img1, size_t stride1, const uint8_t* img2, size_t stride2, size_t width, size_t height, double threshold, int includeAA) {
  // maximum acceptable square distance between two colors;
  // 35215 is the maximum possible value for the YIQ difference metric
  double maxDelta = 35215 * threshold * threshold;
  uint64_t diff   = 0;

    // Parallelize the loop using OpenMP
  #pragma omp parallel for reduction(+ : diff)
  for (int index = 0; index < width * height; index++) {
    // Calculate x and y coordinates from the index
    int y = index / width;
    int x = index % width;
    // allow input images to include different padding in their strides
    int pos1 = y * stride1 + x * 4;
    int pos2 = y * stride2 + x * 4;

    // squared YUV distance between colors at this pixel position
    double delta = color_delta(img1, img2, pos1, pos2, 0);

    // the color difference is above the threshold
    if (delta > maxDelta) {
      // check it's a real rendering difference or just anti-aliasing
      if (includeAA || !(pixelmatch_is_antialiased(img1, x, y, width, height, img2) || pixelmatch_is_antialiased(img2, x, y, width, height, img1))) {
        diff++;
      }
    }
  }

  // return the number of different pixels
  return diff;
}
#else
uint64_t pixelmatch(const uint8_t* img1, size_t stride1, const uint8_t* img2, size_t stride2, size_t width, size_t height, double threshold, int includeAA) {
  // maximum acceptable square distance between two colors;
  // 35215 is the maximum possible value for the YIQ difference metric
  double maxDelta = 35215 * threshold * threshold;
  uint64_t diff   = 0;

  // compare each pixel of one image against the other one
  for (size_t index = 0; index < width * height; index++) {
    // Calculate x and y coordinates from the index
    size_t y = index / width;
    size_t x = index % width;
    // allow input images to include different padding in their strides
    size_t pos1 = y * stride1 + x * 4;
    size_t pos2 = y * stride2 + x * 4;

    // squared YUV distance between colors at this pixel position
    double delta = color_delta(img1, img2, pos1, pos2, 0);

    // the color difference is above the threshold
    if (delta > maxDelta) {
      // check it's a real rendering difference or just anti-aliasing
      if (includeAA || !(pixelmatch_is_antialiased(img1, x, y, width, height, img2) || pixelmatch_is_antialiased(img2, x, y, width, height, img1))) {
        diff++;
      }
    }
  }

  // return the number of different pixels
  return diff;
}
#endif

uint64_t pixelmatch(const uint8_t* img1, const uint8_t* img2, size_t width, size_t height, double threshold, int includeAA) {
  return pixelmatch(img1, width * 4, img2, width * 4, width, height, threshold, includeAA);
}

#endif