#pragma once

#include "mandelbrot.hpp"

namespace mandelbrot_visualizer {

class SequentialMandelbrot : public Mandelbrot {
 public:
  explicit SequentialMandelbrot(const Settings& settings)
      : Mandelbrot(settings) {}

  std::optional<MandelbrotData> Compute(
      const std::atomic<bool>& request_stop) override {
    if (progress) *progress = 0;
    for (int y = 0; y < height; ++y) {
      if (progress)
        *progress = static_cast<float>(y) / static_cast<float>(height);
      for (int x = 0; x < width; ++x) {
        if (request_stop) return std::nullopt;
        std::complex<double> c = PixelToComplex(x, y, width, height, area);
        pixels[y * width + x] = MandelbrotColor(c);
      }
    }
    if (progress) *progress = 1;
    return std::make_optional<MandelbrotData>({height, width, pixels});
  }
};

}  // namespace mandelbrot_visualizer
