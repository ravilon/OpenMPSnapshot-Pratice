#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <omp.h>
#include "Filter.hpp"
#include "Convolution.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace std;

int w_size, bias;     // kernel window size
double** kernel; // image kernel 
vector<Filter*> filters;

// Histogram array to count occurrences of each pixel value 
int histogram[256]; 


// Function to push a filter to the filters vector
void push_filter(int idx) {     
  // make edge detector for color idx 
  double ***ed = get_tensor(w_size, w_size, 3); 
  for (int i = 0; i < w_size; ++i) {
    for (int j = 0; j < w_size; ++j) {
      ed[i][j][idx] = kernel[i][j];
    }
  }
  Filter *f = new Filter(ed, w_size, 3, bias);
  f->normalize();
  filters.push_back(f);
}

// A function to plot the histogram
void plot_histogram(const int* histogram) {
    std::vector<int> x(256), y(256);
    for (int i = 0; i < 256; ++i) {
        x[i] = i;
        y[i] = histogram[i];
    }
    plt::bar(x, y);
    plt::title("Histogram");
    plt::xlabel("Pixel Value");
    plt::ylabel("Frequency");
    plt::save("output/histogram.png"); // Save the plot as an image file
    plt::show(); // Display the plot
}

int main(int argc, char *argv[]) {

  // Initialize histogram to zero
  fill(begin(histogram), end(histogram), 0);

  //  input the kernel matrix
  cin >> w_size >> bias;
  kernel = new double*[w_size];

  for (int i = 0; i < w_size; i++) {

    kernel[i] = new double[w_size];

    for (int j = 0; j < w_size; j++) {

      cin >> kernel[i][j]; 

    }
  }

 // We have 3 filters for R, G, B channels
  push_filter(0); // R
  push_filter(1); // G
  push_filter(2); // B




  if (argc < 3) {
    cerr << "Usage <input_data file name> <output file name>" << endl;
    return 1;
  }
  // output file will be written in the same format as input file
  ifstream ifile (argv[1]);   
  ofstream ofile (argv[2]);
  if (!ofile.is_open()) {
    cerr << "Unable to open " << argv[1] << endl;
    return 1;
  }

  if (!ifile.is_open()) {
    cerr << "Unable to open " << argv[0] << endl;
    return 1;
  }

  int width, num_images, height, depth;

  // As requested in the assignment, we will use a stride of 1
  // As an option feature, we will also use padding of 1
  int stride = 1, padding = 1; 


  ifile >> num_images >> width >> height >> depth;
  
  ofile << num_images << " "; 
  cerr << num_images << " "; 
  
  Convolution clayer(width, height, depth, w_size, stride, padding, filters.size()); 

  double ***input;

  input = get_tensor(width, height, depth); 


  // THIS CODE ONLY RUNS A THREAD FOR EACH IMAGE
  /*#pragma omp parallel for // Parallelize the loop
  
  for (int id = 0; id < num_images; ++id) {

    // read one image
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < height; ++j) {
        for (int k = 0; k < depth; ++k) {
          ifile >> input[i][j][k];
          if (ifile.peek() == ',') ifile.ignore();
        }
      }
    }
    auto output = clayer.conv2d(input, filters);

    double ***out_volume = get<3>(output);

    int o_width = get<0>(output), o_height = get<1>(output), o_depth = get<2>(output);

    #pragma omp critical // Critical section
    {
      if (id == 0) {
        // print image dimensions only the first time
        ofile << o_width << " " << o_height << " " << o_depth << "\n";
        cerr << o_width << " " << o_height << " " << o_depth << "\n";
      }

      for (int i = 0; i < o_width; ++i) {
        for (int j = 0; j < o_height; ++j) {

          // ofile << out_volume[i][j][0] << "," << out_volume[i][j][1] 
          //         << "," << out_volume[i][j][2] << " ";      
          int value[3];
          for (int d = 0; d < 3; ++d)
          {
              value[d] = static_cast<int>(out_volume[i][j][d]);
              value[d] = max(0, min(255, value[d])); // Clamp to [0, 255]
              histogram[value[d]]++; // Increment histogram
          }
          ofile << value[0] << "," << value[1] << "," << value[2] << " ";
        }
        ofile << "\n";
      }
      ofile << "\n";
    }


  }*/


  // THIS CODE RUNS 5 THREADS FOR EACH IMAGE
    for (int id = 0; id < num_images; ++id) {
      // Read one image
      for (int i = 0; i < width; ++i) {
          for (int j = 0; j < height; ++j) {
              for (int k = 0; k < depth; ++k) {
                  ifile >> input[i][j][k];
                  if (ifile.peek() == ',') ifile.ignore();
              }
          }
      }
      
      // Parallelize processing within each image
      /*
        The following #pragma omp parallel num_threads(5) directive creates a parallel region with 5 threads for each image.
        Within this parallel region, the clayer.conv2d(input, filters) function is called. 
        Each thread in the parallel region will independently execute this function, 
        but it's important to ensure that the function's implementation supports thread safety.
      */

      #pragma omp parallel num_threads(5)
      {
        auto output = clayer.conv2d(input, filters);
        double ***out_volume = get<3>(output);
        int o_width = get<0>(output), o_height = get<1>(output), o_depth = get<2>(output);


        /*
          The following #pragma omp critical directive ensures that only one thread at a time can execute the critical section of code. 
          This is important when writing to the output 
        */
        #pragma omp critical
        {
            if (id == 0) {
                ofile << o_width << " " << o_height << " " << o_depth << "\n";
                cerr << o_width << " " << o_height << " " << o_depth << "\n";
            }

            for (int i = 0; i < o_width; ++i) {
                for (int j = 0; j < o_height; ++j) {
                    int value[3];
                    for (int d = 0; d < 3; ++d) {
                        value[d] = static_cast<int>(out_volume[i][j][d]);
                        value[d] = max(0, min(255, value[d])); // Clamp to [0, 255]
                        histogram[value[d]]++; // Increment histogram
                    }
                    ofile << value[0] << "," << value[1] << "," << value[2] << " ";
                }
                ofile << "\n";
            }
            ofile << "\n";
        }
      }
    }
  
  // Write the histogram to a file
  ofstream hfile("output/histogram.txt");
  if (hfile.is_open())
  {
      for (int i = 0; i < 256; ++i)
      {
          hfile << i << " " << histogram[i] << "\n";
      }
      hfile.close();
  }
  else
  {
    cerr << "Unable to open histogram.txt" << endl;
  }
  
  ifile.close();
  ofile.close();

  // Plot the histogram
  plot_histogram(histogram);

  return 0; 
}