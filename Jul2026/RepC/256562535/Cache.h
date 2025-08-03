/*
Author: https://www.geeksforgeeks.org/lru-cache-implementation/
*/

#pragma once
#include <list>
#include <unordered_map>


class Cache {
// store keys of cache 
std::list<int32_t> dq;

// store references of key in cache 
std::unordered_map<int32_t*, std::list<int32_t>::iterator> ma;
int csize; // maximum capacity of cache 

int cacheMiss = 0;
int cacheHit = 0;

public:
Cache(int);
void refer(int32_t*);
void display();
void displayMisses();
};
