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

float xi_lmk(int l, int m, int k) { return 0; }

void compute_c_nlm(int n, int l, int m,
                   sycl::buffer<sycl::vec<float, 4>, 2> &alpha_bl,
                   sycl::buffer<sycl::vec<float, 4>, 3> &beta_nbl,
                   sycl::buffer<sycl::vec<float, 4>, 1> &ps, float sigma,
                   sycl::buffer<cdouble, 3> out, sycl::queue &queue) {
  queue
      .submit([&](sycl::handler &cgh) {
        auto alpha_acc = alpha_bl.get_access<sycl::access::mode::read>(cgh);
        auto beta_acc = beta_nbl.get_access<sycl::access::mode::read>(cgh);
        auto ps_acc = ps.get_access<sycl::access::mode::read>(cgh);
        auto out_acc = out.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for(out.get_range(), [=](sycl::id<3> idx) {
          int n = idx[0];
          int l = idx[1];
          int m = idx[2];

          size_t N_b = alpha_acc.get_range()[1];
          size_t N_p = ps_acc.get_range()[0];

          cdouble c_val_total(0.0f, 0.0f);

          for (size_t b = 0; b < N_b; ++b) {
            double ab = alpha_acc[l][b].x();
            double bb = beta_acc[l][n][b].x();
            double denom =
                std::pow(std::sqrt(1 + 2 * ab * sigma * sigma), 2 * l + 3);
            double factor_b = bb / denom;

            cdouble sum_p_total(0.0f, 0.0f);

            for (size_t p = 0; p < N_p; ++p) {
              double rx = ps_acc[p].x();
              double ry = ps_acc[p].y();
              double rz = ps_acc[p].z();
              double rp = std::sqrt(rx * rx + ry * ry + rz * rz);

              double exp_factor =
                  std::exp(-ab * rp * rp / (1 + 2 * ab * sigma * sigma));
              cdouble xy_complex(std::pow(rx, m), std::pow(ry, m));
              double rp_l_minus_m = (rp != 0) ? std::pow(rp, l - m) : 0.0f;

              double sum_k = 0.0f;
              for (int k = m; k <= l; ++k) {
                double xi_val = xi_lmk(l, m, k);
                sum_k += xi_val * std::pow(rz, k - m) * std::pow(rp, m - k);
              }

              sum_p_total += exp_factor * xy_complex * rp_l_minus_m * sum_k;
            }

            c_val_total += factor_b * sum_p_total * std::pow(-1.0f, m);
          }

          double numerator = std::tgamma(l - m + 1);
          double denominator = 4.0f * M_PI * std::tgamma(l + m + 1);
          double lambda_lm =
              std::pow(2.0f, l) * std::sqrt(numerator / denominator);

          c_val_total *=
              lambda_lm * std::pow(2.0f * sigma * sigma * M_PI, 1.5f);

          out_acc[idx] = cdouble{c_val_total.real(), c_val_total.imag()};
        });
      })
      .wait();
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
  sycl::buffer<sycl::vec<float, 4>, 3> beta_bl_buf{
      sycl::range<3>(beta_bl_shape[0], beta_bl_shape[1], beta_bl_shape[2])};

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
    int aisle = beta_bl_shape[2];
    for (int r = 0; r < row; ++r)
      for (int c = 0; c < col; ++c)
        for (int a = 0; a < aisle; ++a) {
          int idx = r * (col * aisle) + c * aisle + a;
          beta_bl_acc[sycl::id<3>(r, c, a)] = beta_bl_vec[idx];
        }
  }

  size_t XI_L = l_max + 1;

  auto xi_lmk_size = (XI_L + 1) * (XI_L + 2) * (XI_L) / 6;

  sycl::buffer<float, 1> xi_lmk_buf{xi_lmk_size};

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

  cnpy::npy_save("xi_lmk_cpp.npy", xi);

  sycl::buffer<cdouble, 3> c_nlm_buf{
      sycl::range<3>(n_max, l_max + 1, l_max + 1)};

  ScopedTimer timer_c_nlm = "Computing c_nlm";
  queue
      .submit([&](sycl::handler &cgh) {
        auto alpha_acc = alpha_bl_buf.get_access<sycl::access::mode::read>(cgh);
        auto beta_acc = beta_bl_buf.get_access<sycl::access::mode::read>(cgh);
        auto ps_acc = atoms.get_access<sycl::access::mode::read>(cgh);
        auto out_acc = c_nlm_buf.get_access<sycl::access::mode::write>(
            cgh); // write each element once

        cgh.parallel_for(c_nlm_buf.get_range(), [=](sycl::id<3> idx) {
          int n = idx[0];
          int l = idx[1];
          int m = idx[2];

          size_t N_b = alpha_acc.get_range()[1];
          size_t N_p = ps_acc.get_range()[0];

          std::complex<double> c_val_total(0.0f, 0.0f);

          // Sum over basis functions
          for (size_t b = 0; b < N_b; ++b) {
            double ab = alpha_acc[l][b].x();
            double bb = beta_acc[l][n][b].x();
            double denom =
                std::pow(std::sqrt(1 + 2 * ab * sigma * sigma), 2 * l + 3);
            double factor_b = bb / denom;

            cdouble sum_p_total(0.0f, 0.0f);

            // Sum over particle positions
            for (size_t p = 0; p < N_p; ++p) {
              double rx = ps_acc[p].x();
              double ry = ps_acc[p].y();
              double rz = ps_acc[p].z();
              double rp = std::sqrt(rx * rx + ry * ry + rz * rz);

              double exp_factor =
                  std::exp(-ab * rp * rp / (1 + 2 * ab * sigma * sigma));
              cdouble xy_complex(rx, ry);
              xy_complex = std::pow(xy_complex, m); // correct complex power
              double rp_l_minus_m = (rp != 0) ? std::pow(rp, l - m) : 0.0f;

              double sum_k = 0.0f;
              for (int k = m; k <= l; ++k) {
                double xi_val = xi_lmk(l, m, k); // needs actual xi
                sum_k += xi_val * std::pow(rz, k - m) * std::pow(rp, m - k);
              }

              sum_p_total += exp_factor * xy_complex * rp_l_minus_m * sum_k;
            }

            c_val_total += factor_b * sum_p_total * std::pow(-1.0f, m);
          }

          double numerator = std::tgamma(l - m + 1);
          double denominator = 4.0f * M_PI * std::tgamma(l + m + 1);
          double lambda_lm =
              std::pow(2.0f, l) * std::sqrt(numerator / denominator);

          c_val_total *=
              lambda_lm * std::pow(2.0f * sigma * sigma * M_PI, 1.5f);

          out_acc[idx] = cdouble{c_val_total.real(), c_val_total.imag()};
        });
      })
      .wait();
  timer_c_nlm.finish();

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

  // Shape for npy_save: (n_max, l_max+1, l_max+1, 2)
  std::vector<size_t> c_nlm_shape = {(size_t)n_max, (size_t)l_max + 1,
                                     (size_t)l_max + 1, 2};

  cnpy::npy_save("c_nlm_cpp.npy", c_nlm_flat.data(), c_nlm_shape);

  return 0;
}
