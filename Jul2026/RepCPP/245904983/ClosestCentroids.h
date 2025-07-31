#pragma once

#include "headers/Matrix.h"

template<typename T>
class ClosestCentroids : public Matrix<int>{
public:

    /**
     * if buffered ClosestCentroids desired -> 2 rows else 1
    */
    ClosestCentroids(int samples, int value = 0, bool buffered=true, int num_threads = 1) : 
        Matrix<int>((buffered?2:1), samples, value, num_threads),
        _toggle{ (buffered?1:0) } { 
        
        initDistBuffer();
    }
    /**
     * Gets closest cluster index w.r.t. each sample
     * 
     * Dimensions:
     *      rows: 1 (2 if toggle feature activated)
     *      cols: n_samples
    */
    ClosestCentroids& getClosest(const Matrix<T>& data, const Matrix<T>& cluster){
        int n_dims = data.getRows();
        int n_clusters = cluster.getCols();

        #pragma omp parallel for collapse(1) num_threads(_n_threads)
        for(int i = 0; i < _cols; ++i){
            T abs_sum = 0;
            for(int d = 0; d < n_dims; ++d){
                abs_sum += std::abs(data(d, i) - cluster(d, 0));
            }
            _matrix[i+_toggled_row*_cols] = 0;
            (*_distBuffer)(0, i) = abs_sum;
            for(int c = 1; c < n_clusters; ++c){
                abs_sum = 0;
                for(int d = 0; d < n_dims; ++d){
                    abs_sum += std::abs(data(d, i) - cluster(d, c));
                }
                if(abs_sum < (*_distBuffer)(0, i)){
                    _matrix[i+_toggled_row*_cols] = c;
                    (*_distBuffer)(0, i) = abs_sum;
                }
            }
        }
        _current_row = _toggled_row;
        _toggled_row ^= _toggle;
        return *this;
    }

    /**
     * Checks whether the stopping criterion is satisfied or not.
     * If 2 consecutive closest centroids computation's modification
     * rate that is below the given threshold, we consider that 
     * KMeans has converged 
    */
    float getModifRate(){
        // stopping criterion never satisfied if we dont keep track of assigned centroids modifications
        if(_rows < 2) return 1.0f;
        int counter = 0;
        #pragma omp parallel for simd reduction(+:counter) num_threads(_n_threads)
        for(int i = 0; i < _cols; ++i){
            const int& a = _matrix[i];
            const int& b = _matrix[i+_cols];
            if(!(a ^ b)) ++counter;
        }
        return 1.0f - static_cast<float>(counter) / _cols;   
    }

    inline int& operator()(const int& col) { return _matrix[col+_current_row*_cols]; }
    inline const int& operator()(const int& col) const { return _matrix[col+_current_row*_cols]; }

private:
    void initDistBuffer(){
        _distBuffer = std::make_unique<Matrix<T>>(1, _cols, 0, _n_threads);
    }

    std::unique_ptr<Matrix<T>> _distBuffer;
    // we want to store the previous state of mapped centroids, toggle: 1 to switch between rows
    int _toggle = 0;
    // we store the toggled row
    int _toggled_row = 0;
    int _current_row = 0;
};