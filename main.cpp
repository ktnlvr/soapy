#include "cnpy.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>

#include "buffer.hpp"
#include "misc.hpp"
#include "state.hpp"
#include "timer.hpp"

int main(void) {
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
  timer_upload.finish();

  timer_init.finish();

  {
    ScopedTimer timer = "Computing";
    std::vector<Buffer> buffers = {xi_lmk};
    dispatch_compute("xi_lmk", state.boilerplate, shader, buffers, {},
                     l_max + 1, l_max + 1, l_max + 1);
  }

  auto computed = read_buffer(state.boilerplate, xi_lmk);
  cnpy::npy_save("xi_lmk_cpp.npy", computed);

  return 0;
}
