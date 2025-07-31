#pragma once

#include <chrono>
#include <functional>
namespace mandelbrot_visualizer {

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

struct Stopwatch {
  static milliseconds Time(const std::function<void()>& fn) {
    auto start = high_resolution_clock::now();
    fn();
    auto stop = high_resolution_clock::now();
    return duration_cast<milliseconds>(stop - start);
  }
};
}  // namespace mandelbrot_visualizer
