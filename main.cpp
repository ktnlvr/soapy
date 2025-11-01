#include "cnpy.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>

#include "timer.hpp"
#include "misc.hpp"
#include "state.hpp"
#include "buffer.hpp"

int main(void) {
  {
    ScopedTimer timer = "Loading";
    auto [alpha_bl, alpha_bl_shape] = load_numpy_array("alpha_bl.npy");
    auto [beta_bl, beta_bl_shape] = load_numpy_array("beta_nbl.npy");
  }

  int r_cut = 50;
  int l_max = 400;
  int n_max = 2;

  State state = {r_cut, l_max, n_max};
  auto buffer_size = sizeof(float) * (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;

  auto shader = load_shader("xi_lmk.spv");

  auto buffer = create_buffer(state.boilerplate, buffer_size);
  {
    ScopedTimer timer = "Computing";
    dispatch_compute("xi_lmk", state.boilerplate, shader, buffer, l_max + 1, l_max + 1, l_max + 1);
  }
  auto computed = read_buffer(state.boilerplate, buffer);

  cnpy::npy_save("xi_lmk_cpp.npy", computed);

  return 0;
}
