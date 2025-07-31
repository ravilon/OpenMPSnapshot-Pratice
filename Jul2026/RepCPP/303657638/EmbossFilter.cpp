#include "opencv2/opencv.hpp"
#include "EmbossFilter.hpp"
#include <omp.h>

using namespace cv;

/**********************************************************
 *************** EfficientPixelAccess *********************
 *********************************************************/

uchar convolutePixelEfficient(Mat &inputImage, const Mat &kernel, int I, int J, int K)
{
  int kSize = kernel.rows;
  int halfSize = kSize / 2;

  double newPixelValue = 0;

  // go over kernel
  for (int i = 0; i < kSize; i++)
  {
    for (int j = 0; j < kSize; j++)
    {
      // apply kernel on current pixel an his 3x3 neigborhood
      double inputImagePixelValue = static_cast<double>(inputImage.ptr<Vec3b>(I + i - halfSize, J + j - halfSize)->val[K]);
      newPixelValue += inputImagePixelValue * kernel.at<float>(i, j);
    }
  }
  return static_cast<uchar>(min(255, max(0, int(round(newPixelValue)))));
}

void applyEmbossFilterEfficientPixelAccess(Mat &inputImage, Mat &outputImage)
{
  // create an emboss kernel
  float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
  Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

  // initialize the empty output image
  outputImage = Mat::zeros(inputImage.size(), inputImage.type());

  Vec3b *outputImagePointer;

  // go over the image
  for (int i = 1; i < inputImage.rows - 1; i++)
  {
    for (int j = 1; j < inputImage.cols - 1; j++)
    {
      // obtain a pointer for outputImage
      outputImagePointer = outputImage.ptr<Vec3b>(i, j);

      // go over all 3 channels
      for (int k = 0; k < inputImage.channels(); k++)
      {
        outputImagePointer->val[k] = convolutePixelEfficient(inputImage, embossKernel, i, j, k);
      }
    }
  }
}

/**********************************************************
 ************************ Parallel ************************
 *********************************************************/

void applyParallelEmbossFilter(Mat &inputImage, Mat &outputImage)
{
  // create an emboss kernel
  float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
  Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

  // initialize the empty output image
  outputImage = Mat::zeros(inputImage.size(), inputImage.type());

  Vec3b *outputImagePointer;

// go over the image
// parallelize it via OpenMP
// use private(variableX) to ensure each thread will have it's own variableX, else it would result in interferences between the threads
// #pragma omp parallel for collapse(2) private(outputImagePointer)
#pragma omp parallel for private(outputImagePointer)
  for (int i = 1; i < inputImage.rows - 1; i++)
  {
    for (int j = 1; j < inputImage.cols - 1; j++)
    {
      // obtain a pointer for outputImage
      outputImagePointer = outputImage.ptr<Vec3b>(i, j);

      // go over all 3 channels
      for (int k = 0; k < inputImage.channels(); k++)
      {
        outputImagePointer->val[k] = convolutePixelEfficient(inputImage, embossKernel, i, j, k);
      }
    }
  }
}

/**********************************************************
 ****************** SlowPixelAccess ***********************
 *********************************************************/

uchar convolutePixel(const Mat &inputImage, const Mat &kernel, int I, int J, int K)
{
  int kSize = kernel.rows;
  int halfSize = kSize / 2;

  double pixelValue = 0;

  // go over kernel
  for (int i = 0; i < kSize; i++)
  {
    for (int j = 0; j < kSize; j++)
    {
      // apply kernel on current pixel an his 3x3 neigborhood
      uchar inputImagePixel = inputImage.at<Vec3b>(I + i - halfSize, J + j - halfSize)[K];
      pixelValue += static_cast<double>(inputImagePixel) * kernel.at<float>(i, j);
    }
  }
  return static_cast<uchar>(min(255, max(0, int(round(pixelValue)))));
}

void applyEmbossFilterSlowPixelAccess(const Mat &inputImage, Mat &outputImage)
{
  // create an emboss kernel
  float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
  Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

  // initialize the empty output image
  outputImage = Mat::zeros(inputImage.size(), inputImage.type());

  // go over the image
  for (int i = 1; i < inputImage.rows - 1; i++)
  {
    for (int j = 1; j < inputImage.cols - 1; j++)
    {
      // go over all 3 channels
      for (int k = 0; k < inputImage.channels(); k++)
      {
        outputImage.at<Vec3b>(i, j)[k] = convolutePixel(inputImage, embossKernel, i, j, k);
      }
    }
  }
}
