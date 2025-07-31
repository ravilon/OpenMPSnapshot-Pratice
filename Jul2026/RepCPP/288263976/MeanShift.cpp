//
// Created by francesca on 06/10/20.
//

#include <vector>
#include <iostream>
#include <functional>   // std::plus
#include <algorithm>
#include "Point.h"
#include "MeanShift.h"
#include "MeanShiftUtils.h"


Point MeanShift::updatePoint(Point &point, const std::vector<Point> &original_points, const std::string kernel_type="gaussian"){
    std::vector<float> numerator;
//    initialize numerator
//    numerator.reserve(point.getDim());
    for(int i=0; i<point.getDim(); i++){
        numerator.push_back(0);
    }

    float denominator = .0;
    for(auto& orig_point: original_points){
//        numerator
        float distance = computeDistance(point, orig_point);
        float w = computeKernel(distance, this->bandwidth, kernel_type);
//        update position of point to be shifted

        for(int d=0; d<point.getDim(); d++){ //for each dimension d do: num += coordinate_value * w
            numerator[d] += orig_point.getValues()[d] * w;
        }
//        denominator
        denominator+= w;
    }
    std::transform(numerator.begin(), numerator.end(), numerator.begin(),
                   bind2nd(std::divides<float>(), denominator)); // numerator/ denominator
    return Point(numerator);
};

std::vector<Point> MeanShift::doMeanShift(const std::vector<Point> &points, const int num_threads){
//    std::cout << "Mean Shift function" << '\n';
    std::vector<Point> copied_points = points;

    float bandwidth = this->bandwidth;

    for(int i=0; i<this->max_iter; i++){
//#pragma omp parallel for default(none) shared(points, bandwidth, copied_points) schedule(static) num_threads(num_threads)
#pragma omp parallel for default(none) shared(points, bandwidth, copied_points) num_threads(num_threads)
//      iterate over points
        for(int c=0; c< points.size(); c++){
            Point newPoint = updatePoint(copied_points[c], points);
            copied_points[c] = newPoint;
        }
    }
    return copied_points;
}

std::vector<Point> MeanShift::doSeqMeanShift(const std::vector<Point> &points){
//    std::cout << "Mean Shift function" << '\n';
    std::vector<Point> copied_points = points;

    float bandwidth = this->bandwidth;

    for(int i=0; i<this->max_iter; i++){
//      iterate over points
        for(int c=0; c< points.size(); c++){
            Point newPoint = updatePoint(copied_points[c], points);
            copied_points[c] = newPoint;
        }
    }
    return copied_points;
}
