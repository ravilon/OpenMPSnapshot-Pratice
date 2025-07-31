#include "opencv2/opencv.hpp"
#include "RgbToGrayscale.hpp"
#include <omp.h>

using namespace cv;


/**********************************************************
 *************** EfficientPixelAccess *********************
 *********************************************************/

void RgbToGrayscaleEfficientPixelAccess(Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels for RGB!");
  }

  // check that the output image has only a single channel
  if (outputImage.channels() > 1)
  {
    throw("Image has to much channels for Grayscale!");
  }

  // prepare an output image of inputImage size with 1 channel
  outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

  int grayPixelValue;

  Vec3b *inputImagePointer;
  uchar *outputImagePointer;

  // go over the imagess
  for (int i = 0; i < inputImage.rows; i++)
  {
    for (int j = 0; j < inputImage.cols; j++)
    {
      // obtain a pointer for inputImage and another one for outputImage
      inputImagePointer = inputImage.ptr<Vec3b>(i, j);
      outputImagePointer = outputImage.ptr<uchar>(i, j);

      // get R, G and B values
      double R = static_cast<double>(inputImagePointer->val[2]);
      double G = static_cast<double>(inputImagePointer->val[1]);
      double B = static_cast<double>(inputImagePointer->val[0]);

      // calculate grayPixelValue
      grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

      // set grayPixelValue to outputImagePointer
      *outputImagePointer = grayPixelValue;
    }
  }
}

/**********************************************************
 *********************** Parallel *************************
 *********************************************************/

void RgbToGrayscaleParallel(Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels for RGB!");
  }

  // check that the output image has a single channel
  if (outputImage.channels() > 1)
  {
    throw("Image has to much channels for Grayscale!");
  }

  // prepare an output image of inputImage size with 1 channel
  outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

  int grayPixelValue;

  Vec3b *inputImagePointer;
  uchar *outputImagePointer;

// go over the inputImage
// use OpenMP directive to parallelize it
// use private(variableX) to ensure each thread will have it's own variableX, else it would result in interferences between the threads
// #pragma omp parallel for collapse(2) private(inputImagePointer, outputImagePointer, grayPixelValue)
#pragma omp parallel for private(inputImagePointer, outputImagePointer, grayPixelValue)
  for (int i = 0; i < inputImage.rows; i++)
  {
    for (int j = 0; j < inputImage.cols; j++)
    {
      // obtain a pointer for inputImage and another one for outputImage
      inputImagePointer = inputImage.ptr<Vec3b>(i, j);
      outputImagePointer = outputImage.ptr<uchar>(i, j);

      // get R, G and B values
      double R = static_cast<double>(inputImagePointer->val[2]);
      double G = static_cast<double>(inputImagePointer->val[1]);
      double B = static_cast<double>(inputImagePointer->val[0]);

      // calculate grayPixelValue
      grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

      // set grayPixelValue to outputImagePointer
      *outputImagePointer = grayPixelValue;
    }
  }
}

/**********************************************************
 ****************** SlowPixelAccess ***********************
 *********************************************************/

void RgbToGrayscaleSlowPixelAccess(const Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels for RGB!");
  }

  // check that the output image has a single channel
  if (outputImage.channels() > 1)
  {
    throw("Image has to much channels for Grayscale!");
  }

  // prepare an output image of inputImage size with 1 channel
  outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

  Vec3b bgr_pixel;
  int grayPixelValue;

  // go over inputImage
  for (int i = 0; i < inputImage.rows; i++)
  {
    for (int j = 0; j < inputImage.cols; j++)
    {
      // get BGR pixel
      bgr_pixel = inputImage.at<Vec3b>(i, j);

      // get R, G and B values
      double R = static_cast<double>(bgr_pixel[2]);
      double G = static_cast<double>(bgr_pixel[1]);
      double B = static_cast<double>(bgr_pixel[0]);

      // calculate grayPixelValue
      grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

      // set grayPixelValue
      outputImage.at<uchar>(i, j) = grayPixelValue;
    }
  }
}
