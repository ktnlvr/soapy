#include "cnpy.h"
#include <cstring>
#include <iostream>
#include <kompute/Kompute.hpp>
#include <memory>
#include <numeric>
#include <ranges>

#include "timer.hpp"
#include "misc.hpp"

struct State {
  kp::Manager mgr;

  int r_cut = 50;
  int n_max = 2;
  int l_max = 3;

  std::vector<std::shared_ptr<kp::Memory>> xi_lmk_params;
  std::shared_ptr<kp::TensorT<float>> xi_lmk_table;
  std::shared_ptr<kp::Algorithm> xi_lmk_algo;
  std::vector<uint32_t> xi_lmk_spirv;

  State(int r_cut, int n_max, int l_max)
      : mgr(), r_cut(r_cut), n_max(n_max), l_max(l_max) {
    auto xi_lmk_spirv = load_shader("xi_lmk.spv");
  }

  auto prepare_xi_table() {
    {
      ScopedTimer timer = "preparing xi table";
      auto sz = (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;
      std::vector<float> output(sz, std::nanf("0"));

      xi_lmk_table = mgr.tensor(output);
      xi_lmk_params = {xi_lmk_table};

      xi_lmk_algo =
          mgr.algorithm(xi_lmk_params, xi_lmk_spirv,
                        kp::Workgroup({uint32_t(l_max + 1), uint32_t(l_max + 1),
                                       uint32_t(l_max + 1)}));
    }
  }

  auto precompute_xi_table() {
    {
      ScopedTimer timer = "precomputing xi table";
      mgr.sequence()
          ->record<kp::OpSyncDevice>(xi_lmk_params)
          ->record<kp::OpAlgoDispatch>(xi_lmk_algo)
          ->eval();
    }
  }
};

int main(void) {
  {
    ScopedTimer timer = "Loading";
    auto [alpha_bl, alpha_bl_shape] = load_numpy_array("alpha_bl.npy");
    auto [beta_bl, beta_bl_shape] = load_numpy_array("beta_nbl.npy");
  }

  int r_cut = 50;
  int n_max = 2;
  int l_max = 3;

  auto state = State(r_cut, n_max, l_max);
  state.prepare_xi_table();
  state.precompute_xi_table();

  return 0;
}
