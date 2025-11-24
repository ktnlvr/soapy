#include "cnpy.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>

auto size_from_shape(std::vector<size_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

auto load_numpy_array(std::string_view str)
    -> std::pair<std::vector<double>, std::vector<size_t>> {  // <- double
    cnpy::NpyArray arr = cnpy::npy_load(str.data());
    double* data = arr.data<double>();                        // <- double
    std::vector<size_t> shape = arr.shape;
    auto sz = size_from_shape(shape);

    std::vector<double> out(sz);
    std::memcpy(out.data(), data, sizeof(double) * sz);       // <- copy correct size
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
