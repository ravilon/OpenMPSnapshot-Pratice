#pragma once

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "mode.hpp"
#include "state.hpp"

namespace mandelbrot_visualizer::ui {

class Menu {
 public:
  static void ShowMenu(VisualizerState& state);

 private:
  static void ModeCombo(Mode& current);
  static void FpsInfo();
  static void DurationInfo(VisualizerState& state);
  static void WindowInfo(VisualizerState& state);
  static void SetMaxIterations(int& current);
  static void InteractionInfo();
};

}  // namespace mandelbrot_visualizer::ui
