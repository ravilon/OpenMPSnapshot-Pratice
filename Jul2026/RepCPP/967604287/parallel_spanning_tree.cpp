#include <vector>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Class to represent a permutation (vertex/node in the bubble-sort network)
class Permutation {
public:
    vector<uint8_t> elements;

    Permutation() {}
    Permutation(const vector<uint8_t>& elems) : elements(elems) {}

    string toString() const {
        stringstream ss;
        for (size_t i = 0; i < elements.size(); i++) {
            ss << static_cast<int>(elements[i]);
        }
        return ss.str();
    }

    // Get the position of a specific value in the permutation
    int getPosition(uint8_t value) const {
        for (size_t i = 0; i < elements.size(); i++) {
            if (elements[i] == value) {
                return i;
            }
        }
        return -1; // Not found
    }

    // Get the rightmost misplaced position (r(v))
    int getRightmostMisplacedPosition() const {
        for(int i = 0; i < elements.size() - 1; i++){
            if(elements[i] > elements[i+1])
                return elements[i];
        }
        return -1; 
    }

    bool operator==(const Permutation& other) const {
        return elements == other.elements;
    }

    bool operator<(const Permutation& other) const {
        return elements < other.elements;
    }

    // Static method to create an identity permutation of size n
    static Permutation createIdentity(uint8_t n) {
        Permutation p;
        p.elements.resize(n);
        for (uint8_t i = 0; i < n; i++) p.elements[i] = i + 1;
        return p;
    }
};

// Enable hash function for Permutation for unordered_map
namespace std {
    template <>
    struct hash<Permutation> {
        size_t operator()(const Permutation& p) const {
            size_t seed = p.elements.size();
            for (auto& i : p.elements) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

// Swap function as defined in the algorithm
Permutation Swap(const Permutation& v, uint8_t x) {
    int i = v.getPosition(x);  // v^{-1}(x)
    if (i < 0 || i + 1 >= static_cast<int>(v.elements.size()))
        return v;  // Cannot swap (x not found or x is at last position)

    Permutation p = v;
    swap(p.elements[i], p.elements[i + 1]);  // vhii (swap x with next element)
    return p;
}

// FindPosition function
Permutation FindPosition(const Permutation& v, uint8_t t, uint8_t n, const Permutation& identity) {
    uint8_t v_n_1 = v.elements[n - 2];  // v_{n-1}
    uint8_t v_n = v.elements[n - 1];    // v_n

    // (1.1) Special case: t == 2 and Swap(v, t) == identity
    if (t == 2) {
        Permutation swapped = Swap(v, t);
        if (swapped == identity) {
            return Swap(v, t-1);
        }
    }
    // (1.2) If v_{n-1} is t or n-1, then use r(v)
    if (v_n_1 == t || v_n_1 == n - 1) {
        int j = v.getRightmostMisplacedPosition();  
        if (j >= 0) {
            return Swap(v, j);  // Swap the symbol at position j
        }
    }

    // (1.3) Default case
    return Swap(v, t);
}

// Parent1 function
Permutation Parent1(const Permutation& v, uint8_t t, uint8_t n, const Permutation& identity) {
    uint8_t v_n = v.elements[n - 1];     // v_n
    uint8_t v_n_1 = v.elements[n - 2];   // v_{n-1}

    if(v == identity)
        return v;

    if (v_n == n) {
        if (t != n - 1) {
            return FindPosition(v, t, n, identity); // Rule 1
        } else {
            return Swap(v, v_n_1); // Rule 2
        }
    } else if (v_n == n - 1 && v_n_1 == n && !(Swap(v, n) == identity)) {
        if (t == 1) {
            return Swap(v, n);  // Rule 3
        } else {
            return Swap(v, t - 1);  // Rule 4
        }
    } else {
        if (v_n == t) {
            return Swap(v, n);  // Rule 5
        } else {
            return Swap(v, t);  // Rule 6
        }
    }
}

// Function to save tree information (node and parent only)
void saveTreeInfo(uint8_t tree_index, const vector<Permutation>& all_permutations, uint8_t n, const Permutation& identity) {
    string filename = "tree" + to_string(tree_index) + ".txt";
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    file << "Spanning Tree " << static_cast<int>(tree_index) << endl;
    file << "Node\tParent" << endl;

    // Create a map from permutation to index for fast lookup
    unordered_map<Permutation, size_t> perm_to_index;
    for (size_t i = 0; i < all_permutations.size(); i++) {
        perm_to_index[all_permutations[i]] = i;
    }

    // Calculate parents using OpenMP for parallelization
    vector<size_t> parent_indices(all_permutations.size());
    size_t root_idx = 0;

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < all_permutations.size(); i++) {
            // Calculate parent
            Permutation parent = Parent1(all_permutations[i], tree_index, n, identity);
            
            // If this is the root (parent is itself)
            if (all_permutations[i] == parent) {
                root_idx = i;
                parent_indices[i] = i; // Parent of root is itself
            } else {
                // Find index of the parent
                parent_indices[i] = perm_to_index[parent];
            }
        }
    }

    // Write results to file
    for (size_t i = 0; i < all_permutations.size(); i++) {
        const Permutation& node = all_permutations[i];
        const Permutation& parent = all_permutations[parent_indices[i]];
        
        file << node.toString() << "\t"
             << parent.toString() << endl;
    }

    file.close();
}

int main(int argc, char* argv[]) {
    int mpi_rank, mpi_size, provided;

    // Initialize MPI with thread support for OpenMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Check for adequate thread support
    if (provided < MPI_THREAD_FUNNELED && mpi_rank == 0) {
        printf("Warning: MPI implementation does not provide required thread level\n");
    }

    // Set OpenMP threads based on environment or default
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    uint8_t n;

    // Only master process handles input
    if (mpi_rank == 0) {
        cout << "Enter the size of permutations (n): ";
        int input;
        cin >> input;
        
        if (input <= 1 || input > 255) {
            cout << "Please enter a value between 2 and 255." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        n = static_cast<uint8_t>(input);
    }

    // Broadcast n to all processes
    MPI_Bcast(&n, 1, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    // Generate all permutations (all processes need all permutations)
    auto start_permutation = high_resolution_clock::now();

    vector<Permutation> all_permutations;
    vector<uint8_t> initial_perm(n);
    for (uint8_t i = 0; i < n; i++) {
        initial_perm[i] = i + 1; // 1-indexed permutation
    }

    do {
        all_permutations.push_back(Permutation(initial_perm));
    } while (next_permutation(initial_perm.begin(), initial_perm.end()));

    auto end_permutation = high_resolution_clock::now();
    auto duration_permutation = duration_cast<milliseconds>(end_permutation - start_permutation);

    if (mpi_rank == 0) {
        cout << "Total number of permutations (nodes): " << all_permutations.size() << endl;
        cout << "Time to generate all permutations: " << duration_permutation.count() << " milliseconds" << endl;
    }

    // Create identity permutation
    Permutation identity = Permutation::createIdentity(n);

    // Each process handles a subset of trees
    uint8_t num_trees = n - 1;
    auto start_all_trees = high_resolution_clock::now();

    // Distribute trees among processes
    for (uint8_t t = 1; t <= num_trees; t++) {
        // Simple round-robin distribution
        if ((t - 1) % mpi_size == mpi_rank) {
            auto start_tree = high_resolution_clock::now();
            
            if (mpi_rank == 0) { // Only rank 0 prints progress
                cout << "\nGenerating Tree " << static_cast<int>(t) << "..." << endl;
            }
            
            saveTreeInfo(t, all_permutations, n, identity);
            
            auto end_tree = high_resolution_clock::now();
            auto tree_duration = duration_cast<milliseconds>(end_tree - start_tree);
            
            cout << "Process " << mpi_rank << " - Time to generate Tree " 
                 << static_cast<int>(t) << ": " << tree_duration.count() 
                 << " milliseconds" << endl;
        }
    }

    // Wait for all processes to complete
    MPI_Barrier(MPI_COMM_WORLD);

    auto end_all_trees = high_resolution_clock::now();
    auto duration_all_trees = duration_cast<milliseconds>(end_all_trees - start_all_trees);

    if (mpi_rank == 0) {
        cout << "\nTotal time to generate all trees: " << duration_all_trees.count() << " milliseconds" << endl;
        cout << "\nSuccessfully generated " << static_cast<int>(num_trees) << " independent spanning trees." << endl;
    }

    MPI_Finalize();
    return 0;
}
