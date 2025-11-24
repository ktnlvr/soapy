#include <sycl/sycl.hpp>

#include "atoms.hpp"
#include "misc.hpp"
#include "timer.hpp"
#include <cmath>
#include <complex>

int xi_lmk_offset(int l, int m, int k) {
  int offset_l = (l * (l + 1) * (l + 2)) / 6;
  int offset_m = (m * (2 * l - m + 3)) / 2;
  int offset_k = k - m;
  return offset_l + offset_m + offset_k;
}

using cdouble = std::complex<double>;

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

  std::ifstream positions_file("random_hydrogens.xyz");
  std::vector<float> positions = read_atoms_as_vec4(positions_file);
  positions_file.close();

  std::vector<sycl::vec<float, 4>> atom_vec;
  atom_vec.reserve(positions.size() / 4);
  for (size_t i = 0; i < positions.size(); i += 4) {
    atom_vec.push_back(sycl::vec<float, 4>(positions[i], positions[i + 1],
                                           positions[i + 2], positions[i + 3]));
  }

  sycl::buffer<sycl::vec<float, 4>, 1> atoms(atom_vec.data(), atom_vec.size());

  ScopedTimer timer_ab = "Loading alpha/beta nbl";
  auto [alpha_bl_vec, alpha_bl_shape] = load_numpy_array("alpha_bl.npy");
  auto [beta_bl_vec, beta_bl_shape] = load_numpy_array("beta_nbl.npy");
  timer_ab.finish();

  sycl::buffer<double, 2> alpha_bl_buf{
      sycl::range<2>(alpha_bl_shape[0], alpha_bl_shape[1])};
  {
    auto acc = alpha_bl_buf.get_access<sycl::access::mode::write>();
    for (size_t i = 0; i < alpha_bl_shape[0]; ++i)
      for (size_t j = 0; j < alpha_bl_shape[1]; ++j)
        acc[i][j] =
            alpha_bl_vec[i * alpha_bl_shape[1] + j]; // C-order flattening
  }

  sycl::buffer<double, 3> beta_bl_buf{
      sycl::range<3>(beta_bl_shape[0], beta_bl_shape[1], beta_bl_shape[2])};
  {
    auto acc = beta_bl_buf.get_access<sycl::access::mode::write>();
    for (size_t n = 0; n < beta_bl_shape[0]; ++n)
      for (size_t l = 0; l < beta_bl_shape[1]; ++l)
        for (size_t b = 0; b < beta_bl_shape[2]; ++b)
          acc[n][l][b] = beta_bl_vec[n * beta_bl_shape[1] * beta_bl_shape[2] +
                                     l * beta_bl_shape[2] + b];
  }

  size_t XI_L = l_max + 1;

  auto xi_lmk_size = (XI_L + 1) * (XI_L + 2) * (XI_L) / 6;

  sycl::buffer<double, 1> xi_lmk_buf{xi_lmk_size};

  ScopedTimer timer_xi_lmk = "xi_lmk";
  queue
      .submit([&](sycl::handler &cgh) {
        auto xi_acc = xi_lmk_buf.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(
            sycl::range<3>{XI_L, XI_L, XI_L}, [=](sycl::id<3> gid) {
              int l = gid[0];
              int m = gid[1];
              int k = gid[2];
              int offset = xi_lmk_offset(l, m, k);

              if (m > l || k < m || k > l) {
                return;
              }

              double value = 0.0;
              if (l == 0 && m == 0 && k == 0) {
                value = 1.0;
              } else if ((k - l) % 2 == 0) {
                double num =
                    std::tgamma((double(l) + double(k) - 1) / 2.0 + 1.0);
                double den = std::tgamma(k - m + 1.0) *
                             std::tgamma(l - k + 1.0) *
                             std::tgamma((double(l) + double(k) - 1) / 2.0 -
                                         double(l) + 1.0);
                value = num / den;
              } else {
                value = 0.;
              }

              xi_acc[offset] = value;
            });
      })
      .wait();
  timer_xi_lmk.finish();

  std::vector<float> xi(xi_lmk_size);
  {
    auto host_acc = xi_lmk_buf.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < xi_lmk_size; ++i)
      xi[i] = host_acc[i];
  }

  cnpy::npy_save("xi_lmk_cpp.npy", xi);

  sycl::buffer<cdouble, 3> c_nlm_buf{
      sycl::range<3>(n_max, l_max + 1, l_max + 1)};

  ScopedTimer timer_c_nlm = "Computing c_nlm";
  queue
      .submit([&](sycl::handler &cgh) {
        auto alpha_acc = alpha_bl_buf.get_access<sycl::access::mode::read>(cgh);
        auto beta_acc = beta_bl_buf.get_access<sycl::access::mode::read>(cgh);
        auto ps_acc = atoms.get_access<sycl::access::mode::read>(cgh);
        auto out_acc = c_nlm_buf.get_access<sycl::access::mode::write>(cgh);
        auto xi_acc = xi_lmk_buf.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(c_nlm_buf.get_range(), [=](sycl::id<3> idx) {
          int n = idx[0];
          int l = idx[1];
          int m = idx[2];

          size_t N_b = alpha_acc.get_range()[1];
          size_t N_p = ps_acc.get_range()[0];

          std::complex<double> c_val_total(0.0, 0.0);

          for (size_t b = 0; b < N_b; ++b) {
            double ab = alpha_acc[l][b];
            double bb = beta_acc[l][n][b];
            double denom =
                std::pow(std::sqrt(1.0 + 2.0 * ab * sigma * sigma), 2 * l + 3);
            double factor_b = bb / denom;

            std::complex<double> sum_p_total(0.0, 0.0);

            for (size_t p = 0; p < N_p; ++p) {
              double rx = ps_acc[p].x();
              double ry = ps_acc[p].y();
              double rz = ps_acc[p].z();
              double rp = std::sqrt(rx * rx + ry * ry + rz * rz);

              double exponent = ab * rp * rp / (1.0 + 2.0 * ab * sigma * sigma);
              double exp_factor =
                  (exponent < 700.0) ? std::exp(-exponent) : 0.0;

              std::complex<double> xy_complex(1.0, 0.0);
              if (m != 0 || std::abs(rx) > 1e-12 || std::abs(ry) > 1e-12)
                xy_complex = std::pow(std::complex<double>(rx, ry), m);

              double rp_l_minus_m = (rp == 0.0) ? 0.0 : std::pow(rp, l - m);

              double sum_k = 0.0;
              for (int k = m; k <= l; ++k) {
                double xi_val = xi_acc[xi_lmk_offset(l, m, k)];
                double term_k = (rp == 0.0 && m != k)
                                    ? 0.0
                                    : std::pow(rz, k - m) * std::pow(rp, m - k);
                sum_k += xi_val * term_k;
              }

              sum_p_total += exp_factor * xy_complex * rp_l_minus_m * sum_k;
            }

            c_val_total += factor_b * sum_p_total * ((m & 1) ? -1.0 : 1.0);
          }

          double numerator = (2.0 * l + 1.0) * std::tgamma(l - m + 1);
          double denominator = 4.0 * M_PI * std::tgamma(l + m + 1);
          double lambda_lm =
              std::pow(2.0, l) * std::sqrt(numerator / denominator);

          c_val_total *=
              lambda_lm * std::pow(std::sqrt(2.0 * sigma * sigma * M_PI), 3);

          out_acc[idx] = c_val_total;
        });
      })
      .wait();

  std::cerr << timer_c_nlm.finish() << std::endl;

  std::cout << "alpha_bl_buf:" << std::endl;
  for (int l = 0; l < alpha_bl_shape[0]; ++l)
    for (int b = 0; b < alpha_bl_shape[1]; ++b)
      std::cout << alpha_bl_vec[l * alpha_bl_shape[1] + b] << " ";
  std::cout << std::endl;

  std::cout << "beta_bl_buf:" << std::endl;
  for (int n = 0; n < beta_bl_shape[0]; ++n)
    for (int l = 0; l < beta_bl_shape[1]; ++l)
      for (int b = 0; b < beta_bl_shape[2]; ++b)
        std::cout << beta_bl_vec[n * beta_bl_shape[1] * beta_bl_shape[2] +
                                 l * beta_bl_shape[2] + b]
                  << " ";
  std::cout << std::endl;

  std::vector<double> c_nlm_flat(n_max * (l_max + 1) * (l_max + 1) * 2);

  {
    auto host_acc = c_nlm_buf.get_access<sycl::access::mode::read>();
    for (int n = 0; n < n_max; ++n)
      for (int l = 0; l <= l_max; ++l)
        for (int m = 0; m <= l; ++m) {
          int base_idx = ((n * (l_max + 1) + l) * (l_max + 1) + m) * 2;
          cdouble val = host_acc[sycl::id<3>(n, l, m)];
          c_nlm_flat[base_idx + 0] = val.real();
          c_nlm_flat[base_idx + 1] = val.imag();
        }
  }

  std::vector<size_t> c_nlm_shape = {(size_t)n_max, (size_t)l_max + 1,
                                     (size_t)l_max + 1, 2};

  cnpy::npy_save("c_nlm_cpp.npy", c_nlm_flat.data(), c_nlm_shape);

  return 0;
}
