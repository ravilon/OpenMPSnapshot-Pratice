#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "equalizer.hpp"
using namespace std;

HistogramEqualizer::HistogramEqualizer(const std::string& imagePath) :
inputImage(imagePath), outputImage(inputImage.width, inputImage.height) {}

HistogramEqualizer::~HistogramEqualizer() {}

void HistogramEqualizer::saveImage(const std::string& filename) {
outputImage.save(filename);
}

SequentialResult HistogramEqualizer::equalizerSequential() {
SequentialResult result;
auto start = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();

// 1. Computing histogram: counting gray levels occurrences
start = std::chrono::high_resolution_clock::now();
inputImage.histogram.counts.assign(256, 0);                    // Resetting histogram to zero
for(int i = 0; i < inputImage.width * inputImage.height; i++) {
unsigned char pixelValue = inputImage.pixels[i];
inputImage.histogram.counts[pixelValue]++;
}
end = std::chrono::high_resolution_clock::now();
result.seqHistTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// 2. Computing Cumulative Distribution Function (CDF) of the histogram
start = std::chrono::high_resolution_clock::now();
inputImage.histogram.cdf[0] = inputImage.histogram.counts[0];        // first CDF value is equal to first histogram value
for(int i = 1; i < 256; i++) {
inputImage.histogram.cdf[i] = inputImage.histogram.cdf[i-1] + inputImage.histogram.counts[i];
}
end = std::chrono::high_resolution_clock::now();
result.seqCdfComputeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// 3. Normalizing CDF to the range 0-255
start = std::chrono::high_resolution_clock::now();
// finding the first non-zero CDF value, to prevent division by zero
int cdfMin = 0;
for(int i = 0; i < 256; i++) {
if(inputImage.histogram.cdf[i] > 0) {
cdfMin = inputImage.histogram.cdf[i];
break;
}
}
// Normalization formula: (cdf(v) - cdfMin) * (L-1)/(N - cdfMin), with L=256 (gray levels) e N=total number of pixels
for(int i = 0; i < 256; i++) {
inputImage.histogram.cdf[i] = round((float)(inputImage.histogram.cdf[i] - cdfMin) * 255.0 / (inputImage.width * inputImage.height - cdfMin));
}
end = std::chrono::high_resolution_clock::now();
result.seqCdfNormalizeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// 4. Transforming Image
start = std::chrono::high_resolution_clock::now();
for(int i = 0; i < inputImage.width * inputImage.height; i++) {
// taking the old pixel value (that of the input image) and finding the new one using the CDF
unsigned char oldValue = inputImage.pixels[i];
outputImage.pixels[i] = static_cast<unsigned char>(inputImage.histogram.cdf[oldValue]);
}
end = std::chrono::high_resolution_clock::now();
result.seqTransformTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

result.seqExecutionTime = result.seqHistTime + result.seqCdfComputeTime + result.seqCdfNormalizeTime + result.seqTransformTime;

return result;
}

ParallelResult HistogramEqualizer::equalizerParallel(int numThreads, int blockSize) {
ParallelResult result;
result.numThreads = numThreads;
result.blockSize = blockSize;

int totalPixels = inputImage.width * inputImage.height;
vector<int> cdf = inputImage.histogram.cdf;

auto start = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();

// 1. Computing histogram: counting gray levels occurrences
start = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(numThreads)
{
vector<int> local_hist(256, 0);    // local histogram for each thread
#pragma omp for
for (int i = 0; i < totalPixels; i++) {
local_hist[inputImage.pixels[i]]++;
}
#pragma omp critical
{
for(int i = 0; i < 256; i++) {
inputImage.histogram.counts[i] += local_hist[i];
}
}
}
end = std::chrono::high_resolution_clock::now();
result.parHistTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// 2. Computing Cumulative Distribution Function (CDF) of the histogram
start = std::chrono::high_resolution_clock::now();
inputImage.histogram.cdf[0] = inputImage.histogram.counts[0];
for(int i = 1; i < 256; i++) {
inputImage.histogram.cdf[i] = inputImage.histogram.cdf[i-1] + inputImage.histogram.counts[i];
}
end = std::chrono::high_resolution_clock::now();
result.parCdfComputeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// 3. Normalizing CDF to the range 0-255
start = std::chrono::high_resolution_clock::now();
int cdfMin = 0;
for(int i = 0; i < 256; i++) {
if(inputImage.histogram.cdf[i] > 0) {
cdfMin = inputImage.histogram.cdf[i];
break;
}
}
for(int i = 0; i < 256; i++) {
inputImage.histogram.cdf[i] = round((float)(inputImage.histogram.cdf[i] - cdfMin) * 255.0 / (inputImage.width * inputImage.height - cdfMin));
}
end = std::chrono::high_resolution_clock::now();
result.parCdfNormalizeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// 4. Transforming Image
start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(numThreads)
for(int i = 0; i < totalPixels; i++) {
outputImage.pixels[i] = static_cast<unsigned char>(cdf[inputImage.pixels[i]]);
}
end = std::chrono::high_resolution_clock::now();
result.parTransformTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

result.parExecutionTime = result.parHistTime + result.parCdfComputeTime + result.parCdfNormalizeTime + result.parTransformTime;

return result;
}
