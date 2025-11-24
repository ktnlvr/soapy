#include <sycl/sycl.hpp>

#include "atoms.hpp"
#include "misc.hpp"
#include "timer.hpp"
#include <cmath>

int xi_lmk_offset(int l, int m, int k) {
  int offset_l = (l * (l + 1) * (l + 2)) / 6;
  int offset_m = (m * (2 * l - m + 3)) / 2;
  int offset_k = k - m;
  return offset_l + offset_m + offset_k;
}

int main() {
  for (const auto &platform : sycl::platform::get_platforms()) {
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
              << "\n";

    for (const auto &device : platform.get_devices()) {
      std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                << " | Type: ";

      auto type = device.get_info<sycl::info::device::device_type>();
      switch (type) {
      case sycl::info::device_type::cpu:
        std::cout << "CPU";
        break;
      case sycl::info::device_type::gpu:
        std::cout << "GPU";
        break;
      case sycl::info::device_type::accelerator:
        std::cout << "Accelerator";
        break;
      default:
        std::cout << "Other";
      }

      std::cout << "\n";
    }

    std::cout << std::endl;
  }

  sycl::queue queue;
  auto device = queue.get_device();

  // Warmup kernel
  queue.submit([&](sycl::handler &cgh) { cgh.single_task([=]() {}); }).wait();

  std::cout << "Running on: " << device.get_info<sycl::info::device::name>()
            << " ("
            << (device.get_info<sycl::info::device::device_type>() ==
                        sycl::info::device_type::gpu
                    ? "GPU"
                : device.get_info<sycl::info::device::device_type>() ==
                        sycl::info::device_type::cpu
                    ? "CPU"
                    : "Other")
            << ")" << std::endl;

  if (device.has(sycl::aspect::fp64)) {
    std::cout << "Double precision supported\n";
  } else {
    std::cout << "Double precision NOT supported, exiting\n";
    exit(-1);
  }

  int r_cut = 50;
  int l_max = 3;
  int n_max = 2;
  float sigma = 1.;

  sycl::buffer<sycl::vec<float, 4>, 1> atoms(4);

  std::ifstream positions_file("random_hydrogens.xyz");
  std::vector<float> positions = read_atoms_as_vec4(positions_file);
  positions_file.close();

  ScopedTimer timer_ab = "Loading alpha/beta nbl";
  auto [alpha_bl_vec, alpha_bl_shape] = load_numpy_array("alpha_bl.npy");
  auto [beta_bl_vec, beta_bl_shape] = load_numpy_array("beta_nbl.npy");
  timer_ab.finish();

  sycl::buffer<sycl::vec<float, 4>, 2> alpha_bl_buf{
      sycl::range<2>(alpha_bl_shape[0], alpha_bl_shape[1])};
  sycl::buffer<sycl::vec<float, 4>, 2> beta_bl_buf{
      sycl::range<2>(beta_bl_shape[0], beta_bl_shape[1])};

  {
    sycl::host_accessor alpha_bl_acc(alpha_bl_buf, sycl::write_only);
    int row = alpha_bl_shape[0];
    int col = alpha_bl_shape[1];
    for (int r = 0; r < row; ++r)
      for (int c = 0; c < col; ++c)
        alpha_bl_acc[sycl::id<2>(r, c)] = alpha_bl_vec[r * col + c];
  }

  {
    sycl::host_accessor beta_bl_acc(beta_bl_buf, sycl::write_only);
    int row = beta_bl_shape[0];
    int col = beta_bl_shape[1];
    for (int r = 0; r < row; ++r)
      for (int c = 0; c < col; ++c)
        beta_bl_acc[sycl::id<2>(r, c)] = beta_bl_vec[r * col + c];
  }

  size_t XI_L = l_max + 1;

  auto xi_lmk_size = (XI_L + 1) * (XI_L + 2) * (XI_L) / 6;

  sycl::buffer<float, 1> xi_lmk_buf{xi_lmk_size};

  sycl::range<3> local_range;

  queue.submit([&](sycl::handler &cgh) {
    auto xi_acc = xi_lmk_buf.get_access<sycl::access::mode::write>(cgh);
    
    cgh.parallel_for(sycl::range<3>{XI_L, XI_L, XI_L}, [=](sycl::id<3> gid) {
      int l = gid[0];
      int m = gid[1];
      int k = gid[2];

      if (m > l || k < m || k > l)
        return;

      float value = 0.0f;
      if (l == 0 && m == 0 && k == 0) {
        value = 1.0f;
      } else if ((k - l) % 2 == 0) {
        double num = std::tgamma((double(l) + double(k) - 1) / 2.0 + 1.0);
        double den =
            std::tgamma(k - m + 1.0) * std::tgamma(l - k + 1.0) *
            std::tgamma((double(l) + double(k) - 1) / 2.0 - double(l) + 1.0);
        value = float(num / den);
      }

      int offset = xi_lmk_offset(l, m, k);
      xi_acc[offset] = value;
    });
  });

  std::vector<float> xi(xi_lmk_size);
  {
    auto host_acc = xi_lmk_buf.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < xi_lmk_size; ++i)
      xi[i] = host_acc[i];
  }

  std::cout << xi.size() << std::endl;

  cnpy::npy_save("xi_lmk_cpp.npy", xi);

  return 0;
}
