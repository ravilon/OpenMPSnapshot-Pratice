/*
This is a class that parses a gedcom file into a structure of 
person objects. Created by Jacob McCoy on 28 November 2023
*/

#include "RelationshipFinder.h"
#include <climits>
#include <sstream>
#include <iostream>
#include <chrono>

// chunk size determines how many iterations of a for loop each thread does
// at a time. change as desired
#define CHUNK_SIZE 25 

// leave defined if you want OpenMP in your code
#define OMP_TEST_MODE

// the constructor creates the adjacency matrix dynamically
RelationshipFinder::RelationshipFinder(std::unordered_map<fs_id, Person> person_map)
{
    // copy in the person map
    this->family_map = person_map;

    // make the dynamic arrays
    this->matrix_width = person_map.size();
    this->adjacency_matrix = new unsigned int[this->matrix_width * this->matrix_width];
    this->prev = new unsigned int[this->matrix_width * this->matrix_width];

    // get the max value
    this->max_dist = UINT_MAX - this->matrix_width - 1;

    // fill it with default values
    auto start_time = std::chrono::high_resolution_clock::now();
    #ifdef OMP_TEST_MODE
    #pragma omp parallel for schedule(guided, CHUNK_SIZE)
    #endif
    for (int i = 0; i < this->matrix_width; i++)
    {
        for (int j = 0; j < this->matrix_width; j++)
        {
            // by default, the distance from one node to another is infinity
            this->adjacency_matrix[i*matrix_width + j] = this->max_dist;
            // by default, to get from i from j, you go directly to i
            this->prev[i*matrix_width + j] = i;
        }
        // // put zeros on the diagonal for the adjacency matrix
        // // and itself on the diagonal for the previous matrix
        this->adjacency_matrix[i*matrix_width + i] = 0;
        this->prev[i*matrix_width + i] = i;
    }

    // print out some timing stats
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    #ifdef OMP_TEST_MODE
    std::cout << "OpenMP ";
    #endif
    std::cout << "Matrix Filling Time: " << duration.count() << " us" << std::endl;

    // for each item in the map
    for (auto& person : person_map)
    {
        // pull out the id
        id current_id = person.second.GetID();

        // on that id's row, add a 1 in the column of each relationship
        // spouse relationships
        for (int i = 0; i < person.second.GetSpouses().size(); i++)
        {
            this->adjacency_matrix[current_id * this->matrix_width + person.second.GetSpouses().at(i)] = 1;
        }
        // child relationships
        for (int i = 0; i < person.second.GetChildren().size(); i++)
        {
            this->adjacency_matrix[current_id * this->matrix_width + person.second.GetChildren().at(i)] = 1;
        }
        // father relationships
        for (int i = 0; i < person.second.GetFathers().size(); i++)
        {
            this->adjacency_matrix[current_id * this->matrix_width + person.second.GetFathers().at(i)] = 1;
        }
        // mother relationships
        for (int i = 0; i < person.second.GetMothers().size(); i++)
        {
            this->adjacency_matrix[current_id * this->matrix_width + person.second.GetMothers().at(i)] = 1;
        }
    }

    // // error checking. matrix should be symmetric
    for (int i = 0; i < this->matrix_width; i++)
    {
        for (int j = 0; j< this->matrix_width; j++)
        {
            if (this->adjacency_matrix[i*this->matrix_width + j] != this->adjacency_matrix[j*this->matrix_width + i])
            { 
                std::cerr << "Error: (" << i << "," << j << ") != (" << j << "," << i << ").";
                std::cerr << "(" << i << "," << j << ")=" << this->adjacency_matrix[i*this->matrix_width + j] << " but ";
                std::cerr << "(" << j << "," << i << ")=" << this->adjacency_matrix[j*this->matrix_width + i] <<"\n";

                // look for the errors
                for (auto it = this->family_map.begin(); it != this->family_map.end(); ++it)
                {
                    if (it->second.GetID() == i)
                    {
                        std::cout << i << ": " <<it->second.GetName() << "fs: " << it->first << std::endl;
                    }
                    if (it->second.GetID() == j)
                    {
                        std::cout << j << ": " <<it->second.GetName() << "fs: " <<it->first << std::endl;
                    }
                }

            }
        }
    }

}

RelationshipFinder::~RelationshipFinder()
{
    // deallocate the dynamic arrays
    delete[] this->adjacency_matrix;
    delete[] this->prev;
}

std::string RelationshipFinder::ToString() const
{
    // format the string as a grid, with x meaning there's no path
    std::stringstream to_return;
    for (int i = 0; i < this->matrix_width; i++)
    {
        for (int j = 0; j < this->matrix_width; j++)
        {   unsigned int to_insert = this->adjacency_matrix[i*this->matrix_width + j];
            if (to_insert == this->max_dist)
            {
                to_return << "x" << " ";
            }
            else
            {
                to_return << to_insert << " ";
            }
            
        }
        to_return << std::endl;
    }
    return to_return.str();
}

std::string RelationshipFinder::ToStringPath() const
{
    // same as ToString(), just uses the path matrix not the distance matrix
    std::stringstream to_return;
    for (int i = 0; i < this->matrix_width; i++)
    {
        for (int j = 0; j < this->matrix_width; j++)
        {   unsigned int to_insert = this->prev[i*this->matrix_width + j];
            if (to_insert == this->max_dist)
            {
                to_return << "x" << " ";
            }
            else
            {
                to_return << to_insert << " ";
            }  
        }
        to_return << std::endl;
    }
    return to_return.str();
}

void RelationshipFinder::FloydRelationshipFinder()
{
    unsigned int matrix_width = this->matrix_width;

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    // go through every node
    for (int k = 0; k < matrix_width; k++)
    {
        #ifdef OMP_TEST_MODE
        #pragma omp parallel for schedule(guided, CHUNK_SIZE)
        #endif
        // for every pair of nodes
        for (int i = 0; i < matrix_width; i++)
        {
            for (int j = 0; j < matrix_width; j++)
            {
                // if the path from i to j is longer than the path from i to j that cuts through k
                if (this->adjacency_matrix[i*matrix_width + j] >
                    this->adjacency_matrix[i*matrix_width + k] +
                    this->adjacency_matrix[k*matrix_width + j])
                    {
                        // make the path that cuts through k be the new path
                        this->adjacency_matrix[i*matrix_width + j] =
                        this->adjacency_matrix[i*matrix_width + k] +
                        this->adjacency_matrix[k*matrix_width + j];

                        // get the path
                        this->prev[i*matrix_width + j] = this->prev[k*matrix_width + j];
                    }
            }
        }
    }

    // print out the timing of everything
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    #ifdef OMP_TEST_MODE
    std::cout << "OpenMP ";
    #endif
    std::cout << "Floyd's Algorithm Time: " << duration.count() << " us" << std::endl;
    return;
}

void RelationshipFinder::DisplayPath(fs_id start, fs_id end)
{
    // check for valid IDs
    bool start_valid = false;
    bool end_valid = false;
    //Probably can parallelize. can't really have a data race I don't think
    for (auto it = this->family_map.begin(); it != this->family_map.end(); ++it)
    {
        // if it isn't true and they're equal...
        if((!start_valid) && it->first == start)
        {
            start_valid = true;
        }
        // do for end point as well
        if((!end_valid) && it->first == end)
        {
            end_valid = true;
        }
    }

    // if one isn't valid, end it all
    if (!(start_valid && end_valid))
    {
        std::cout << "Invalid ID Found" << std::endl;
        return;
    }

    // pull out the indices to use on the matrix
    id start_index, current_index, end_index;
    current_index = start_index = this->family_map[start].GetID();
    end_index = this->family_map[end].GetID();

    // base case: no connection between the two
    if (this->prev[start_index * this->matrix_width + end_index] == this->max_dist)
    {
        std::cout << "No relationship" << std::endl;
        return;
    }
    // otherwise, there is a relationship: print it out
    std::cout << "Relationship: " << std::endl;
    for (auto it = this->family_map.begin(); it != this->family_map.end(); ++it)
    {
        if (it->second.GetID() == current_index)
        {
            std::cout << it->second.GetName() << std::endl;
            break;
        }
    }

    // go until you hit the end index
    while (current_index != end_index)
    {
        // calculate the new index
        current_index = prev[end_index * this->matrix_width + current_index];
        Person person;
        // go find the right person
        for (auto it = this->family_map.begin(); it != this->family_map.end(); ++it)
        {
            if (it->second.GetID() == current_index)
            {
                // print out their name if it's the right one
                std::cout << it->second.GetName() << std::endl;
                break;
            }
        }
    }
    
    return;    
}
