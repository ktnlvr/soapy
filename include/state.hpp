#pragma once

#include "boilerplate.hpp"

struct State {
  Boilerplate boilerplate;
  int r_cut, l_max, n_max;
  
  State(int r_cut, int l_max, int n_max)
      : r_cut(r_cut), l_max(l_max), n_max(n_max) {}
};
