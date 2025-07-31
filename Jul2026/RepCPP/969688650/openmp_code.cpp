// does not use adjacency list
// #include <iostream>
// #include <vector>
// #include <numeric>
// #include <algorithm>
// #include <fstream>
// #include <omp.h>
// #include <cstdlib>
// using namespace std;

// vector<int> indexToPerm(int n, int idx) {
//     vector<int> perm(n);
//     iota(perm.begin(), perm.end(), 1);
//     vector<int> result;
//     for (int i = n; i > 0; i--) {
//         int fact = 1;
//         for (int j = 1; j < i; j++)
//             fact *= j;
//         int pos = idx / fact;
//         result.push_back(perm[pos]);
//         perm.erase(perm.begin() + pos);
//         idx %= fact;
//     }
//     return result;
// }

// int permToIndex(const vector<int>& perm) {
//     int index = 0, n = perm.size();
//     vector<int> temp = perm;
//     for (int i = 0; i < n; ++i) {
//         int smaller = 0;
//         for (int j = i + 1; j < n; ++j)
//             if (temp[j] < temp[i])
//                 smaller++;
//         int fact = 1;
//         for (int j = 1; j < n - i; j++)
//             fact *= j;
//         index += smaller * fact;
//     }
//     return index;
// }

// bool isIdentity(const vector<int>& v) {
//     for (int i = 0; i < v.size(); i++) {
//         if (v[i] != i + 1) return false;
//     }
//     return true;
// }

// int findR(const vector<int>& v) {
//     int n = v.size();
//     for (int i = n - 1; i >= 0; i--) {
//         if (v[i] != i + 1) return i + 1; // +1 for 1-based indexing
//     }
//     return -1;
// }

// vector<int> Swap(const vector<int>& v, int x) {
//     vector<int> res = v;
//     int i = -1;
//     for (int j = 0; j < v.size(); j++) {
//         if (v[j] == x) {
//             i = j;
//             break;
//         }
//     }
//     if (i == -1 || i == v.size() - 1) return res;
//     swap(res[i], res[i + 1]);
//     return res;
// }

// vector<int> FindPosition(const vector<int>& v, int t, int n) {
//     vector<int> one_n(n);
//     iota(one_n.begin(), one_n.end(), 1);

//     if (t == 2 && Swap(v, t) == one_n) {
//         return Swap(v, t - 1);
//     } else if (v[n - 2] == t || v[n - 2] == n - 1) {
//         int j = findR(v);
//         return Swap(v, j);
//     } else {
//         return Swap(v, t);
//     }
// }

// vector<int> Parent1(const vector<int>& v, int t, int n) {
//     vector<int> one_n(n);
//     iota(one_n.begin(), one_n.end(), 1);

//     if (v[n - 1] == n) {
//         if (t != n - 1) {
//             return FindPosition(v, t, n);
//         } else {
//             return Swap(v, v[n - 2]);
//         }
//     } else if (v[n - 1] == n - 1 && v[n - 2] == n && !isIdentity(Swap(v, n))) {
//         if (t == 1) {
//             return Swap(v, n);
//         } else {
//             return Swap(v, t - 1);
//         }
//     } else {
//         if (v[n - 1] == t) {
//             return Swap(v, n);
//         } else {
//             return Swap(v, t);
//         }
//     }
// }

// void writeDotFile(const vector<pair<int, int>>& edges, int treeNum, int n) {
//     ofstream dot("spanning_tree_" + to_string(treeNum) + ".dot");
//     dot << "digraph Tree" << treeNum << " {\n";
//     for (const auto& [child, parent] : edges) {
//         string c = "";
//         for (int x : indexToPerm(n, child)) c += to_string(x);
//         string p = "";
//         for (int x : indexToPerm(n, parent)) p += to_string(x);
//         dot << "    \"" << p << "\" -> \"" << c << "\";\n";
//     }
//     dot << "}\n";
//     dot.close();
// }

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         cerr << "Usage: " << argv[0] << " <number_of_threads>" << endl;
//         return 1;
//     }

//     int num_threads = atoi(argv[1]);
//     if (num_threads <= 0) {
//         cerr << "Number of threads must be positive" << endl;
//         return 1;
//     }
//     omp_set_num_threads(num_threads);

//     int n;
//     cout << "Enter n: ";
//     cin >> n;
//     if (n <= 0) {
//         cerr << "n must be positive" << endl;
//         return 1;
//     }

//     int totalPerms = 1;
//     for (int i = 2; i <= n; ++i)
//         totalPerms *= i;

//     vector<vector<pair<int, int>>> localSpanning(n - 1);

//     #pragma omp parallel for schedule(dynamic)
//     for (int idx = 0; idx < totalPerms; ++idx) {
//         auto perm = indexToPerm(n, idx);
//         for (int t = 1; t < n; ++t) {
//             if (isIdentity(perm))
//                 continue;
//             vector<int> parent_perm = Parent1(perm, t, n);
//             int parent = permToIndex(parent_perm);
//             #pragma omp critical
//             localSpanning[t - 1].emplace_back(idx, parent);
//         }
//     }

//     for (int t = 0; t < n - 1; ++t) {
//         vector<pair<int, int>> edges = localSpanning[t];
//         cout << "\n=== Spanning Tree " << t + 1 << " ===\n";
//         for (const auto& [child, parent] : edges) {
//             for (int x : indexToPerm(n, child)) cout << x;
//             cout << " <-- ";
//             for (int x : indexToPerm(n, parent)) cout << x;
//             cout << "\n";
//         }
//         writeDotFile(edges, t + 1, n);
//     }

//     return 0;
// }

// --- Uses Adjaceny List ---
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <cstdlib>
using namespace std;

vector<int> indexToPerm(int n, int idx) {
    vector<int> perm(n);
    iota(perm.begin(), perm.end(), 1);
    vector<int> result;
    for (int i = n; i > 0; i--) {
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

int permToIndex(const vector<int>& perm) {
    int index = 0, n = perm.size();
    vector<int> temp = perm;
    for (int i = 0; i < n; ++i) {
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

vector<vector<int>> generateAdjList(int n, int totalPerms, vector<int>& xadj, vector<int>& adjncy) {
    xadj.push_back(0);
    for (int i = 0; i < totalPerms; ++i) {
        auto perm = indexToPerm(n, i);
        for (int j = 0; j < n - 1; ++j) {
            auto neighbor = perm;
            swap(neighbor[j], neighbor[j + 1]);
            adjncy.push_back(permToIndex(neighbor));
        }
        xadj.push_back(adjncy.size());
    }
    vector<vector<int>> dummy;
    return dummy; // unused
}

bool isIdentity(const vector<int>& v) {
    for (int i = 0; i < v.size(); i++) {
        if (v[i] != i + 1) return false;
    }
    return true;
}

int findR(const vector<int>& v) {
    int n = v.size();
    for (int i = n - 1; i >= 0; i--) {
        if (v[i] != i + 1) return i + 1; // +1 for 1-based indexing
    }
    return -1;
}

vector<int> Swap(const vector<int>& v, int x) {
    vector<int> res = v;
    int i = -1;
    for (int j = 0; j < v.size(); j++) {
        if (v[j] == x) {
            i = j;
            break;
        }
    }
    if (i == -1 || i == v.size() - 1) return res;
    swap(res[i], res[i + 1]);
    return res;
}

vector<int> FindPosition(const vector<int>& v, int t, int n) {
    vector<int> one_n(n);
    iota(one_n.begin(), one_n.end(), 1);

    if (t == 2 && Swap(v, t) == one_n) {
        return Swap(v, t - 1);
    } else if (v[n - 2] == t || v[n - 2] == n - 1) {
        int j = findR(v);
        return Swap(v, j);
    } else {
        return Swap(v, t);
    }
}

vector<int> Parent1(const vector<int>& v, int t, int n) {
    vector<int> one_n(n);
    iota(one_n.begin(), one_n.end(), 1);

    if (v[n - 1] == n) {
        if (t != n - 1) {
            return FindPosition(v, t, n);
        } else {
            return Swap(v, v[n - 2]);
        }
    } else if (v[n - 1] == n - 1 && v[n - 2] == n && !isIdentity(Swap(v, n))) {
        if (t == 1) {
            return Swap(v, n);
        } else {
            return Swap(v, t - 1);
        }
    } else {
        if (v[n - 1] == t) {
            return Swap(v, n);
        } else {
            return Swap(v, t);
        }
    }
}

void writeDotFile(const vector<pair<int, int>>& edges, int treeNum, int n) {
    ofstream dot("spanning_tree_" + to_string(treeNum) + ".dot");
    dot << "digraph Tree" << treeNum << " {\n";
    for (const auto& [child, parent] : edges) {
        string c = "";
        for (int x : indexToPerm(n, child)) c += to_string(x);
        string p = "";
        for (int x : indexToPerm(n, parent)) p += to_string(x);
        dot << "    \"" << p << "\" -> \"" << c << "\";\n";
    }
    dot << "}\n";
    dot.close();
}

int main(int argc, char* argv[]) {
    omp_set_num_threads(omp_get_max_threads());
    cout << "Using " << omp_get_max_threads() << " threads.\n";

    int n;
    cout << "Enter n: ";
    cin >> n;
    if (n <= 0) {
        cerr << "n must be positive" << endl;
        return 1;
    }

    int totalPerms = 1;
    for (int i = 2; i <= n; ++i)
        totalPerms *= i;

    vector<int> xadj, adjncy;
    generateAdjList(n, totalPerms, xadj, adjncy);

    vector<vector<pair<int, int>>> localSpanning(n - 1);

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < xadj.size() - 1; ++idx) {
        auto perm = indexToPerm(n, idx);
        for (int t = 1; t < n; ++t) {
            if (isIdentity(perm))
                continue;
            vector<int> parent_perm = Parent1(perm, t, n);
            int parent = permToIndex(parent_perm);
            #pragma omp critical
            localSpanning[t - 1].emplace_back(idx, parent);
        }
    }

    for (int t = 0; t < n - 1; ++t) {
        vector<pair<int, int>> edges = localSpanning[t];
        /*cout << "\n=== Spanning Tree " << t + 1 << " ===\n";
        for (const auto& [child, parent] : edges) {
            for (int x : indexToPerm(n, child)) cout << x;
            cout << " <-- ";
            for (int x : indexToPerm(n, parent)) cout << x;
            cout << "\n";
        }
        */
        writeDotFile(edges, t + 1, n);
    }

    return 0;
}