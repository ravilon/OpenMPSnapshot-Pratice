#pragma once

#include "headers/Matrix.h"
#include "ClosestCentroids.h"

template<typename T>
class KMeans{
public:
KMeans(const Matrix<T>& dataset, int n_clusters, bool stop_criterion=true, int n_threads=1);

Matrix<T> getCentroid();
Matrix<int> getDataToCentroid();
int getNIters();

void mapSampleToCentroid();
void updateCentroids();
void run(int max_iter, float threashold=-1);

void print();


private:
bool _stop_crit;
int _n_threads;
int _n_iters = 0;
/**
* number of features of the dataset (x0, x1, ..., xn)
*/
int _dims;
/**
* number of training samples
*/
int _samples;
/**
* desired number of clusters 
*/
int _n_clusters; 
/**
* K cluster centroids µ1, ..., µK. NxK matrix
* where:
*      N: number of dimensions
*      K: number of classes/clusters
*/
std::unique_ptr<Matrix<T>> _centroids;
/**
* By convention, _training_set is a NxM matrix
* where:
*      N: number of dimensions
*      M: number of training samples 
*/
Matrix<T> _training_set;
/**
* M centroids indices mapping each training sample to a
* corresponding cluster. 1xM matrix
* where:
*      M: number of samples
*      element: index of cluster mapping 
*               taining_index -> cluster_index
* note: M may increase as we add new samples
* The class doesn't support that yet
* 
* stopping criterion idea:
*      keep _dataset_cluster from step t-1
*      and check for changes.
*      If almost no change -> stop algorithm
*/
std::unique_ptr<ClosestCentroids<T>> _dataset_to_centroids;
};

template<typename T>
KMeans<T>::KMeans(const Matrix<T>& dataset, int n_clusters, bool stop_criterion, int n_threads) : 
_training_set{ dataset },
_n_clusters{ n_clusters },
_stop_crit{ stop_criterion },
_n_threads{ n_threads } {

_training_set = dataset;
_dims = dataset.getRows();
_samples = dataset.getCols();
_training_set.setThreads(_n_threads);

Matrix<T> vMinValues = _training_set.vMin();
Matrix<T> vMaxValues = _training_set.vMax();

_centroids = std::make_unique<Matrix<T>>(_dims, n_clusters, UNIFORM, vMinValues, vMaxValues);
_centroids->setThreads(_n_threads);

_dataset_to_centroids = std::make_unique<ClosestCentroids<T>>(_samples, 0, stop_criterion, _n_threads);
}

template<typename T>
inline Matrix<T> KMeans<T>::getCentroid(){ return *_centroids; }

template<typename T>
inline Matrix<int> KMeans<T>::getDataToCentroid(){ return *static_cast<Matrix<int>* >(_dataset_to_centroids.get()); }

template<typename T>
inline int KMeans<T>::getNIters(){ return _n_iters; }

template<typename T>
void KMeans<T>::mapSampleToCentroid(){ _dataset_to_centroids->getClosest(_training_set, *_centroids); }

template<typename T>
void KMeans<T>::updateCentroids(){
// number of points assigned to a cluster
int occurences[_n_clusters] = {0};
// accumulates the samples to compute new cluster positions
T sample_buff[_n_clusters*_dims] = {0};

//#pragma omp parallel for num_threads(_n_threads)
for(int i = 0; i < _samples; ++i){
const int& k_index = (*_dataset_to_centroids)(i);
for(int d = 0; d < _dims; ++d){
//#pragma atomic read write
sample_buff[k_index+d*_n_clusters] += _training_set(d, i);
}
//#pragma atomic write
++occurences[k_index];
}
//#pragma omp parallel for num_threads(_n_threads)
for(int c = 0; c < _n_clusters; ++c){
if(!occurences[c]) continue;
for(int d = 0; d < _dims; ++d){
(*_centroids)(d, c) = sample_buff[c+d*_n_clusters] / occurences[c];
}
}
}

template<typename T>
void KMeans<T>::run(int max_iter, float threashold){

mapSampleToCentroid();
updateCentroids();
_n_iters = 1;
if(max_iter == 1) return;
int epoch = 1;
float modif_rate_prev = 0;
float modif_rate_curr;
float inertia;
do {
mapSampleToCentroid();
updateCentroids();
modif_rate_curr = _dataset_to_centroids->getModifRate();
inertia = modif_rate_curr - modif_rate_prev;
modif_rate_prev = modif_rate_curr;
//printf("%.3f %.3f\n", modif_rate_curr, inertia);
++epoch;
} while(epoch < max_iter && modif_rate_curr >= threashold  && std::abs(inertia) >= 1e-2);
//} while(epoch < max_iter && modif_rate_curr > threashold);
//} while(epoch < max_iter);
_n_iters = epoch;
//printf("iter number: %d\n", epoch);
}

template<typename T>
void KMeans<T>::print() {
for(int d = 0; d < _dims; ++d){
std::cout << "[";
std::cout << _centroids->row(d) << "]," << std::endl;
}
}
