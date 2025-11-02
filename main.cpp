#include "cnpy.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>

#include "atoms.hpp"
#include "buffer.hpp"
#include "misc.hpp"
#include "state.hpp"
#include "timer.hpp"

struct Params {
    int n_max;
    int l_max;
    int N_p;
    int N_b;
    float sigma;
};

int main(void) {
  std::ifstream positions_file("random_hydrogens.xyz");
  std::vector<float> positions = read_atoms_as_vec4(positions_file);
  positions_file.close();

  ScopedTimer timer_ab = "Loading alpha/beta nbl";
  auto [alpha_bl, alpha_bl_shape] = load_numpy_array("alpha_bl.npy");
  auto [beta_bl, beta_bl_shape] = load_numpy_array("beta_nbl.npy");
  timer_ab.finish();

  ScopedTimer timer_init = "initialization routine";

  int r_cut = 50;
  int l_max = 3;
  int n_max = 2;
  float sigma = 1.;

  State state = {r_cut, l_max, n_max};
  auto buffer_size =
      sizeof(float) * (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;

  auto xi_lmk_shader = load_shader("xi_lmk.spv");
  auto c_nlm_shader = load_shader("c_nlm.spv");

  ScopedTimer timer_upload = "Creating buffers";

  auto xi_lmk_buffer = create_buffer(state.boilerplate, buffer_size);
  auto alpha_bl_buffer = create_buffer(
      state.boilerplate, size_from_shape(alpha_bl_shape) * sizeof(float),
      alpha_bl.data());
  auto beta_bl_buffer = create_buffer(
      state.boilerplate, size_from_shape(beta_bl_shape) * sizeof(float),
      beta_bl.data());
  auto positions_buffer = create_buffer(state.boilerplate, sizeof(float) * positions.size(), positions.data());

  int N_b = alpha_bl_shape[1];
  int N_p = positions.size() / 4;
  Params params = { n_max, l_max, N_p, N_b, sigma};
  Buffer params_buffer = create_buffer(state.boilerplate, sizeof(Params), &params);

  auto c_nlm_buffer_size = 2 * sizeof(float) * n_max * (l_max + 1) * (l_max + 1);
  auto c_nlm_buffer = create_buffer(state.boilerplate, c_nlm_buffer_size);

  timer_upload.finish();
  
  std::vector<size_t> positions_arr_shape = {positions.size() / 4, 4};
  auto positions_read_back = read_buffer(state.boilerplate, positions_buffer);
  cnpy::npy_save("positions_cpp.npy", positions_read_back.data(), positions_arr_shape);

  timer_init.finish();

  {
    ScopedTimer timer = "Computing xi_lmk";
    std::vector<Buffer> buffers = {xi_lmk_buffer};
    dispatch_compute("xi_lmk", state.boilerplate, xi_lmk_shader, buffers, {},
                     l_max + 1, l_max + 1, l_max + 1);
  }

  {
    ScopedTimer timer = "Computing c_nlm";
    std::vector<Buffer> buffers = {alpha_bl_buffer, beta_bl_buffer, positions_buffer, xi_lmk_buffer, c_nlm_buffer};
    std::unordered_map<uint32_t, Buffer> uniforms = {{5, params_buffer}};
    dispatch_compute("c_nlm", state.boilerplate, c_nlm_shader, buffers, uniforms,
                     n_max, l_max + 1, l_max + 1);
  }

  ScopedTimer timer_download = "Downloading the xi_lmk";
  auto computed = read_buffer(state.boilerplate, xi_lmk_buffer);
  timer_download.finish();
  cnpy::npy_save("xi_lmk_cpp.npy", computed);

  ScopedTimer timer_download_c_nlm_buffer = "Downloading the xi_lmk";
  auto computed_timer_download_c_nlm_buffer = read_buffer(state.boilerplate, c_nlm_buffer);
  timer_download_c_nlm_buffer.finish();
  std::vector<size_t> c_nlm_shape = {(size_t)n_max, (size_t)l_max + 1, (size_t)l_max + 1, 2};
  cnpy::npy_save("c_nlm_cpp.npy", computed_timer_download_c_nlm_buffer.data(), c_nlm_shape);

  return 0;
}
