#include <iostream>
#include <bits/stdc++.h>
#include <omp.h>



using namespace std;

// template function for finding edge with minimum weight
// connecting current MST with rest of Graph
template <typename A ,typename B , typename C>
int minKeyOMP(A key[], B mstSet[],C V){

A min = INT_MAX;
int index, i;
#pragma omp parallel
    {

        int index_local = index;
        A min_local = min;
// Giving each iterate of for loop to threads
#pragma omp for nowait
        for (i = 0; i < V; i++)
        {
            if (mstSet[i] == false && key[i] < min_local)
            {
                min_local = key[i];
                index_local = i;
            }
        }
// Do this part one by one
// It will avoid working on same variable at same time
#pragma omp critical
        {
            if (min_local < min)
            {
                min = min_local;
                index = index_local;
            }
        }
    }
    return index;

}


// Template function to find MST
// We can pass any type of graph adjacency matrixx together with an array
// It will put MST edges in given array
template <typename T , size_t M, size_t N>
void PrimsOMP(T (&graph)[M][N] , int Parent[]){

    int V = M;
    T key[V];
    int u;
    // if the vertex is in MST , we will set it true
    // otherwise we will set it`s value to false
    bool mstSet[V];

    // set parent of root
    Parent[0] = -1;
    key[0] = 0;
    // set keys to maximum
    for(int i = 1 ; i<V ; ++i){
        key[i] = INT_MAX;
        mstSet[i] = false;
    }

    for(int count = 0 ; count < V ; ++count){
        // find the next vertex to be included in MST
        u = minKeyOMP(key,mstSet,V);

        // include next vertex in MST
        mstSet[u] = true;

        // divides the iterations into chunks
#pragma omp parallel for schedule(static)

        // Set parents of each vertex so we can keep track of
        // edges in MST
        // set key of each vertex equal to weight of connecting edge
        for(int v = 0 ; v < V ; ++v){
            if( graph[u][v] < key[v] && mstSet[v] == false && graph[u][v]){
                key[v] = graph[u][v];
                Parent[v] = u;
            }
        }
    }

}



