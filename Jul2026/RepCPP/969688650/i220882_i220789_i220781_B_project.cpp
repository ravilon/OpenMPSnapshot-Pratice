#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include <metis.h>
#include <fstream>
using namespace std;


vector<int> indexToPerm(int n, int idx)
{
    vector<int> perm(n);
    iota(perm.begin(), perm.end(), 1);
    vector<int> result;
    for (int i = n; i > 0; i--)
    {
        int fact = 1;
        for (int j = 1; j < i; j++)
            fact *= j;
        int pos = idx / fact;
        result.push_back(perm[pos]);
        perm.erase(perm.begin() + pos);
        idx %= fact;
    }
    return result;
}

int permToIndex(const vector<int> &perm)
{
    int index = 0, n = perm.size();
    vector<int> temp = perm;
    for (int i = 0; i < n; ++i)
    {
        int smaller = 0;
        for (int j = i + 1; j < n; ++j)
            if (temp[j] < temp[i])
                smaller++;
        int fact = 1;
        for (int j = 1; j < n - i; j++)
            fact *= j;
        index += smaller * fact;
    }
    return index;
}

vector<vector<int>> generateAdjList(int n, int totalPerms, vector<int> &xadj, vector<int> &adjncy)
{
    xadj.push_back(0);
    for (int i = 0; i < totalPerms; ++i)
    {
        auto perm = indexToPerm(n, i);
        for (int j = 0; j < n - 1; ++j)
        {
            auto neighbor = perm;
            swap(neighbor[j], neighbor[j + 1]);
            adjncy.push_back(permToIndex(neighbor));
        }
        xadj.push_back(adjncy.size());
    }
    vector<vector<int>> dummy;
    return dummy; // unused
}

// Helper function to check if permutation is identity (1,2,...,n)
bool isIdentity(const vector<int>& v) {
    for (int i = 0; i < v.size(); i++) {
        if (v[i] != i+1) return false;
    }
    return true;
}

// Find the position of the first symbol from right not in its correct position
int findR(const vector<int>& v) {
    int n = v.size();
    for (int i = n-1; i >= 0; i--) {
        if (v[i] != i+1) return i+1; // +1 to match paper's 1-based indexing
    }
    return -1;
}

// Swap function exactly as in the paper
vector<int> Swap(const vector<int>& v, int x) {
    vector<int> res = v;
    int i = -1;
    // Find position of x in the permutation (1-based)
    for (int j = 0; j < v.size(); j++) {
        if (v[j] == x) {
            i = j;
            break;
        }
    }
    if (i == -1 || i == v.size()-1) return res;
    swap(res[i], res[i+1]);
    return res;
}

// FindPosition function exactly as in the paper
vector<int> FindPosition(const vector<int>& v, int t, int n) {
    vector<int> one_n(n);
    iota(one_n.begin(), one_n.end(), 1);

    if (t == 2 && Swap(v, t) == one_n) {
        return Swap(v, t-1);
    }
    else if (v[n-2] == t || v[n-2] == n-1) {
        // int j = v[n-1];
        int j = findR(v);
        return Swap(v, j);
    }
    else {
        return Swap(v, t);
    }
}

// Parent1 function exactly as in the paper
vector<int> Parent1(const vector<int>& v, int t, int n) {
    vector<int> one_n(n);
    iota(one_n.begin(), one_n.end(), 1);
    
    if (v[n-1] == n) {
        if (t != n-1) {
            return FindPosition(v, t, n);
        }
        else {
            return Swap(v, v[n-2]);
        }
    }
    else if (v[n-1] == n-1 && v[n-2] == n && !isIdentity(Swap(v, n))) {
        if (t == 1) {
            return Swap(v, n);
        }
        else {
            return Swap(v, t-1);
        }
    }
    else {
        if (v[n-1] == t) {
            return Swap(v, n);
        }
        else {
            return Swap(v, t);
        }
    }
}
void writeDotFile(const vector<pair<int, int>> &edges, int treeNum, int n)
{
    ofstream dot("spanning_tree_" + to_string(treeNum) + ".dot");
    dot << "digraph Tree" << treeNum << " {\n";
    for (const auto &[child, parent] : edges)
    {
       // if (child != parent)  // skip self-loops
        {
            string c = "";
            for (int x : indexToPerm(n, child)) c += to_string(x);
            string p = "";
            for (int x : indexToPerm(n, parent)) p += to_string(x);
            dot << "    \"" << p << "\" -> \"" << c << "\";\n";
        }
    }
    dot << "}\n";
    dot.close();
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);


    int n;
    if (rank == 0)
    {
        cout << "Enter n: " << endl;
        cin >> n;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int totalPerms = 1;
    for (int i = 2; i <= n; ++i)
        totalPerms *= i;

    vector<int> xadj, adjncy;
    vector<idx_t> part(totalPerms);

    if (rank == 0)
    {
        generateAdjList(n, totalPerms, xadj, adjncy);
        idx_t nvtxs = totalPerms;
        idx_t ncon = 1, nparts = size, objval;
        METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                            NULL, NULL, NULL, &nparts,
                            NULL, NULL, NULL, &objval, part.data());
    }

    // Broadcast partition result
    MPI_Bcast(part.data(), totalPerms, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute vertex indices
    vector<int> local_verts;
    for (int i = 0; i < totalPerms; ++i)
        if (part[i] == rank)
            local_verts.push_back(i);

    vector<vector<pair<int, int>>> localSpanning(n - 1);

    cout << "Hello from process " << rank << " ";
    cout << "out of " << size << " ";
    cout << "on " << processor_name << "\n";

    #pragma omp parallel for
    for (int i = 0; i < local_verts.size(); ++i) {
        int idx = local_verts[i];
        auto perm = indexToPerm(n, idx);
        for (int t = 1; t < n; ++t) {
            if(isIdentity(perm))
                continue;
            vector<int> parent_perm = Parent1(perm, t, n);
            int parent = permToIndex(parent_perm);
            #pragma omp critical
            localSpanning[t - 1].emplace_back(idx, parent);
        }
    }

    for (int t = 0; t < n - 1; ++t)
    {
        vector<int> flat;
        for (auto &[c, p] : localSpanning[t])
        {
            flat.push_back(c);
            flat.push_back(p);
        }

        int local_size = flat.size();
        vector<int> recvcounts(size), displs(size), all_flat;
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            int total = 0;
            for (int i = 0; i < size; ++i)
            {
                displs[i] = total;
                total += recvcounts[i];
            }
            all_flat.resize(displs[size - 1] + recvcounts[size - 1]);
        }

        MPI_Gatherv(flat.data(), local_size, MPI_INT,
                    all_flat.data(), recvcounts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

                    if (rank == 0)
                    {
                        vector<pair<int, int>> edges;
                       // cout << "\n=== Spanning Tree " << t + 1 << " ===\n";
                        for (int i = 0; i < all_flat.size(); i += 2)
                        {
                            int child = all_flat[i];
                            int parent = all_flat[i + 1];
                    
                           // if (child == parent) continue; // skip self-loops
                    
                            edges.emplace_back(child, parent);
                    /*
                            for (int x : indexToPerm(n, child)) cout << x;
                            cout << " <-- ";
                            for (int x : indexToPerm(n, parent)) cout << x;
                            cout << "\n";
                            */
                        }
                    
                        // Write Graphviz DOT file
                        writeDotFile(edges, t + 1, n);
                    }
    }

    MPI_Finalize();
    return 0;
}