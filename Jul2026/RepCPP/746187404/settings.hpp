#pragma once

#include <complex>
#include <memory>

namespace mandelbrot_visualizer {

struct Settings {
  struct Area {
    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
    std::complex<double> start{-2.0F, -1.0F};
    std::complex<double> end{1.0F, 1.0F};
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers)

    bool operator!=(const Area& other) const {
      return start != other.start || end != other.end;
    }
  };
  int height;
  int width;
  int max_iterations;
  std::shared_ptr<float> progress;
  Area area;
};

;

}  // namespace mandelbrot_visualizer
