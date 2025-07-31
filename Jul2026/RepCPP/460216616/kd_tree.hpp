#ifndef KD_TREE_H
#define KD_TREE_H

// #pragma once

#include <vector>
// #include <ostream>

#define N_DIM 2


// template<typename T, int ndim>
template<typename T>
struct kpoint{
    T points[N_DIM];
    int ndim;

    kpoint();
    kpoint(T &points_) : points(points_), ndim(N_DIM) {}

    T get_n_point(int axis);

    void set_point(int axis, T value);

    void print_kpoints(int axis);

    // friend std::ostream& operator<<(std::ostream& os, const kpoint (&pt)[N_DIM]){

    //     os << "( ";
    //     for(auto i=0; i<ndim; ++i){
    //         os << pt[i] <<", ";
    //     }
    //     os << ") ";
        
    //     return os;
    // }

};

template<typename T>
struct kdnode {
    int axis; // the splitting dimension
    struct kpoint<T> split; // the splitting element
    struct kdnode<T> *left, *right; // the left and right sub-trees

    kdnode();

    void in_order();

};



template<typename T>
struct kdnode<T> * build_kdtree( std::vector<struct kpoint<T>> points, int ndim, int axis );


#endif

