#pragma once

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <functional>
#include <optional>
#include <string>

namespace mandelbrot_visualizer::ui {

struct SelectionArea {
  int start_x{};
  int start_y{};
  int end_x{};
  int end_y{};
};

struct Selection {
  SelectionArea area{};
  bool has_selected{};
  bool selecting{};
};

struct WindowInfo {
  int width{};
  int height{};
  std::optional<SelectionArea> mouse_selection;
  bool reset_area{};
};

class Window {
 public:
  Window(const std::string& name, int width, int height);
  Window(const Window&) = default;
  Window(Window&&) = delete;
  Window& operator=(const Window&) = default;
  Window& operator=(Window&&) = delete;
  ~Window();
  void EndlessRender(const std::function<void()>& imgui_components,
                     const std::function<void(const WindowInfo&)>& on_render);

 private:
  GLFWwindow* window_{};

  void SetupGLFW(int width, int height, const std::string& name);
  void SetupImGui();
};

}  // namespace mandelbrot_visualizer::ui
