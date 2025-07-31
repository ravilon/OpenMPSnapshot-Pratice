#pragma once

#include <opencv2/opencv.hpp>

/**
 * Dilate a binary image with structuring element
 */
void bdilate(const cv::Mat& in, cv::Mat& out, const cv::Mat& se);

/**
 * Erode a binary image with structuring element
 */
void berode(const cv::Mat& in, cv::Mat& out, const cv::Mat& se);
