#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2> void vasinh_core(const F1 &a, F2 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vasinh(y.size(), a.begin(), y.begin(), y.get_device_mem_stat());

  logger.func_out();
}
} // namespace

} // namespace monolish
