#include "cnpy.h"
#include <cstring>
#include <iostream>
#include <kompute/Kompute.hpp>
#include <memory>
#include <numeric>
#include <ranges>

auto load_numpy_array(std::string_view str)
    -> std::pair<std::vector<float>, std::vector<size_t>> {
  cnpy::NpyArray arr = cnpy::npy_load(str.data());
  float *data = arr.data<float>();
  std::vector<size_t> shape = arr.shape;
  auto sz =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<float> out;
  out.resize(sz);
  std::memcpy(out.data(), data, sizeof(float) * sz);
  return std::make_pair(out, shape);
}

std::vector<uint32_t> load_shader(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  size_t size = (size_t)file.tellg();
  std::vector<uint32_t> buffer(size / sizeof(uint32_t));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(buffer.data()), size);
  return buffer;
}
