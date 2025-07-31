#include <iostream>
#include <fstream>
#include <sstream>
#include <climits>      /* INT_MAX*/
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <cmath>
#include <omp.h>
using namespace std;

#ifndef MAXITER
#define MAXITER 500
#endif

#ifndef OUTFILE
#define OUTFILE "out_omp.txt"
#endif

#ifndef THREADNUM
#define THREADNUM omp_get_max_threads()
#endif

double inline dist(double x1, double y1, double x2, double y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  // sqrt is omitted as it reduces performance.
}

/* Selecting k distinct, random centroids from n points. */
void randomCenters(int *&x, int *&y, int n, int k, double *&cx, double *&cy) {
  int *centroids = new int[k];

#ifdef RANDOM
  srand (time(NULL)); //normal code
  int added = 0;
  
  while(added != k) {
    bool exists = false;
    int temp = rand() % n;
    for(int i = 0; i < added; i++) {
      if(centroids[i] == temp) {
        exists = true;
      }
    }
    if(!exists) {
      cx[added] = x[temp];
      cy[added] = y[temp];
      centroids[added++] = temp;
    }
  }
#else //deterministic init
  for(int i = 0; i < k; i++) {
     cx[i] = x[i];
     cy[i] = y[i];
     centroids[i] = i;
  }
#endif
delete[] centroids;
}

bool assign(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
  bool end = true;
  int * changed = new int[n]();

  omp_set_num_threads(THREADNUM);
  #pragma omp parallel for // reduction(&:end) // Parallelize the assignment step
  for(int i = 0; i < n; i++) {
    int cluster;
    int minDist = INT_MAX;
    // Assign to closest center
    for(int j = 0; j < k; j++) {
      double distance = dist(cx[j], cy[j], x[i], y[i]);      
      if(distance < minDist) {
        minDist = distance;
        cluster = j;
      }
    }
    if(cluster != c[i]) { // even if one point changes, we will run for one more iteration.
      // end = false;
      changed[i] = 1;
      c[i] = cluster; // assign the point to the cluster with minDist
    }
    // REDUCTION FOR SOME REASON FAILS MISERABLY
    // even if one point changes, we will run for one more iteration.
    // end &= (cluster != c[i]);
    // c[i] = cluster; // assign the point to the cluster with minDist
  }

  int sum = 0;

  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < n; i++){
    sum += changed[i];
  }

  delete[] changed;
  return (sum == 0); // if sum == 0 -> no changes --> end == true
  // return end;
}

void init(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
  #ifdef ALIGNED_ALLOC
    cx = (double *) aligned_alloc(64, k * sizeof(double));
    cy = (double *) aligned_alloc(64, k * sizeof(double));
  #else
    cx = new double[k];
    cy = new double[k];
  #endif
  
  randomCenters(x, y, n, k, cx, cy);

  /* Initialize the cluster information for each point. */
  #ifdef ALIGNED_ALLOC
    c = (int *) aligned_alloc(64, n * sizeof(int)); // l1,2,3 cache line size == 64 bytes // preventing false sharing
  #else
    c = new int[n];
  #endif
  
  assign(x,y,c,cx,cy,k,n); // Assign each point to closest center.
}

void update(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
    // double sumx[k] = {0.0}, sumy[k] = {0.0};
    // int count[k] = {0};
    double *sumx = new double[k]; double *sumy = new double[k];
    int *count = new int[k];

    #pragma omp parallel for
    for (int i = 0; i < k; i++){
        sumx[i] = 0.0;
        sumy[i] = 0.0;
        count[i] = 0;
    }

    omp_set_num_threads(THREADNUM);
    #pragma omp parallel for reduction(+:sumx[:k],sumy[:k],count[:k])
    for (int pt = 0; pt < n; pt++){
        // for small values of k: access locality into sumx, sumy & count is better since the set of values contained within array c (0 to k-1) is smaller, hence all elts will likely fit into the cache. As k grows larger we wait more
        sumx[c[pt]] += x[pt];
        sumy[c[pt]] += y[pt];
        count[c[pt]]++;
    } // barrier here 

    omp_set_num_threads(( k < THREADNUM ) ? (k) : (THREADNUM)); // max concurrency here is k
    #pragma omp parallel for schedule(static, 1)
    for (int cl = 0; cl < k; cl++){
        cx[cl] = sumx[cl] / count[cl];
        cy[cl] = sumy[cl] / count[cl];
    }
}

int readfile(string fname, int *&x, int *&y) {
  ifstream f;
  f.open(fname.c_str());
  string line;
  getline(f,line);
  int n = atoi(line.c_str());

  #ifdef ALIGNED_ALLOC
    x = (int*) aligned_alloc(64, n * sizeof(int));
    y = (int*) aligned_alloc(64, n * sizeof(int));
  #else
    x = new int[n];
    y = new int[n];
  #endif
  
  int tempx, tempy;
  for(int i = 0; i < n; i++) {
    getline(f,line);
    stringstream ss(line);
   
    ss >> tempx>> tempy;
    x[i] = tempx;
    y[i] = tempy;
  }
  return n;
}

void print(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
  for(int i = 0; i < k; i++) {
    printf("**Cluster %d **",i);
    printf("**Center :(%f,%f)\n",cx[i],cy[i]);
    for(int j = 0; j < n; j++) {
      if(c[j] == i)
        printf("(%d,%d) ",x[j],y[j]);
    }
    printf("\n");
  }
}

void writeClusterAssignments(const int* x, const int* y, const int* c, int n, const string& filename) {
    ofstream outFile(filename);
    if (!outFile) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    for (int i = 0; i < n; i++) {
        outFile << x[i] << " " << y[i] << " " << c[i] << "\n";
    }
    
    outFile.close();
}

void kmeans(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
  bool end = false; 
  int iter = 0;
  while(!end && iter != MAXITER) {
    update(x,y,c,cx,cy,k,n);  // Update the centers
    #ifdef DEBUG
        printf("=============================\n");
        for(int i = 0; i < k; i++) {
            printf("**Cluster %d **",i);
            printf("**Center :(%f,%f)\n",cx[i],cy[i]);
        }
    #endif
    end = assign(x,y,c,cx,cy,k,n);  // Reassign points to clusters
    iter++;
    if(end) {
        printf("End at iter :%d\n",iter);
        writeClusterAssignments(x, y, c, n, OUTFILE);
    }
  }
  printf("Total %d iterations.\n",iter);
}
  
void usage() {
  printf("./test <filename> <k>\n");
  exit(-1);
}

int main(int argc, char *argv[]) {
  
  if(argc - 1 != 2) {
    usage();
  }

  string fname = argv[1];
  int    k     = atoi(argv[2]);
  
  int    *x;  // array of x coordinates
  int    *y;  // array of y coordinates
  double *cx; // array of x coordinates of centers
  double *cy;  // array of y coordinates of centers
  int    *c;  // array of cluster info
  
  int n = readfile(fname,x,y);
  // std::cout << "Max Threads OMP: " << omp_get_max_threads() << std::endl; // 8 on my craptop
  
  // Measure time for initialization
  double init_start = omp_get_wtime();
  init(x, y, c, cx, cy, k, n);
  double init_end = omp_get_wtime();
  printf("Initialization Time: %f seconds\n", init_end - init_start);

  // Measure time for k-means clustering
  double kmeans_start = omp_get_wtime();
  kmeans(x, y, c, cx, cy, k, n);
  double kmeans_end = omp_get_wtime();
  printf("K-Means Execution Time: %f seconds\n", kmeans_end - kmeans_start);
  
  
  // Evaluate clustering quality
  double totalSSD = 0.0;
  for (int i = 0; i < n; i++) {
    int cluster = c[i];
    totalSSD += dist(x[i], y[i], cx[cluster], cy[cluster]);
  }
  printf("Sqrt of Sum of Squared Distances (SSD): %f\n", sqrt(totalSSD));
  
  // Uncomment to print results
  #ifdef DEBUG
  print(x,y,c,cx,cy,k,n);
  #endif

  delete[] x;
  delete[] y;
  delete[] cx;
  delete[] cy;
  delete[] c;

  return 0;
}
