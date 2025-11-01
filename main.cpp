#include "cnpy.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>

#include "timer.hpp"
#include "misc.hpp"
#include "state.hpp"

int main(void) {
  {
    ScopedTimer timer = "Loading";
    auto [alpha_bl, alpha_bl_shape] = load_numpy_array("alpha_bl.npy");
    auto [beta_bl, beta_bl_shape] = load_numpy_array("beta_nbl.npy");
  }

  int r_cut = 50;
  int l_max = 3;
  int n_max = 2;

  State state = {r_cut, l_max, n_max};

  return 0;
}
