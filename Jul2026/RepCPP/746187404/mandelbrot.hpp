#pragma once

#include <imgui.h>

#include <atomic>
#include <complex>
#include <memory>
#include <optional>
#include <vector>

#include "settings.hpp"

namespace mandelbrot_visualizer {

struct Color {
  float r;
  float g;
  float b;
  float a;

  bool operator==(const Color &other) const {
    return (r == other.r) && (g == other.g) && (b == other.b) && (a == other.a);
  }

  bool operator!=(const Color &other) const { return !(*this == other); }
};

struct MandelbrotData {
  int height;
  int width;
  std::vector<Color> pixels;

  bool operator==(const MandelbrotData &other) const {
    return (height == other.height) && (width == other.width) &&
           (pixels == other.pixels);
  }

  bool operator!=(const MandelbrotData &other) const {
    return !(*this == other);
  }
};

class Mandelbrot {
 public:
  explicit Mandelbrot(const Settings &settings)
      : height(settings.height),
        width(settings.width),
        pixels(height * width),
        max_iterations(settings.max_iterations),
        progress(settings.progress),
        area(settings.area) {}
  virtual ~Mandelbrot() = default;
  Mandelbrot(const Mandelbrot &) = default;
  Mandelbrot(Mandelbrot &&) = default;
  Mandelbrot &operator=(const Mandelbrot &) = delete;
  Mandelbrot &operator=(Mandelbrot &&) = delete;

  virtual std::optional<MandelbrotData> Compute(
      const std::atomic<bool> &request_stop) = 0;

  [[nodiscard]] static std::complex<double> PixelToComplex(int x, int y,
                                                           int width,
                                                           int height,
                                                           Settings::Area area);

  [[nodiscard]] Color MandelbrotColor(const std::complex<double> &c) const;

  [[nodiscard]] int Iteration(const std::complex<double> &c) const;

 protected:
  // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes,cppcoreguidelines-avoid-const-or-ref-data-members)
  const int height;
  const int width;
  std::vector<Color> pixels;
  const int max_iterations;
  const std::shared_ptr<float> progress;
  const Settings::Area area;
  // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes,cppcoreguidelines-avoid-const-or-ref-data-members)
};

}  // namespace mandelbrot_visualizer
