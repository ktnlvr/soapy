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

  State state = {r_cut, l_max, n_max};
  auto buffer_size =
      sizeof(float) * (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;

  auto shader = load_shader("xi_lmk.spv");

  ScopedTimer timer_upload = "Creating buffers";
  auto xi_lmk = create_buffer(state.boilerplate, buffer_size);
  auto alpha_bl_buffer = create_buffer(
      state.boilerplate, size_from_shape(alpha_bl_shape) * sizeof(float),
      alpha_bl.data());
  auto beta_bl_buffer = create_buffer(
      state.boilerplate, size_from_shape(beta_bl_shape) * sizeof(float),
      beta_bl.data());
  auto positions_buffer = create_buffer(state.boilerplate, sizeof(float) * positions.size(), positions.data());
  timer_upload.finish();
  
  std::vector<size_t> positions_arr_shape = {positions.size() / 4, 4};
  auto positions_read_back = read_buffer(state.boilerplate, positions_buffer);
  cnpy::npy_save("positions_cpp.npy", positions_read_back.data(), positions_arr_shape);

  timer_init.finish();

  {
    ScopedTimer timer = "Computing";
    std::vector<Buffer> buffers = {xi_lmk};
    dispatch_compute("xi_lmk", state.boilerplate, shader, buffers, {},
                     l_max + 1, l_max + 1, l_max + 1);
  }

  ScopedTimer timer_download = "Downloading the xi_lmk";
  auto computed = read_buffer(state.boilerplate, xi_lmk);
  timer_download.finish();
  cnpy::npy_save("xi_lmk_cpp.npy", computed);

  return 0;
}
