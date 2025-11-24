#pragma once

#include <chrono>
#include <iostream>
#include <string>

class ScopedTimer {
public:
  ScopedTimer(const char *name)
      : name_(name), start_(std::chrono::high_resolution_clock::now()),
        done(false) {}

  ~ScopedTimer() {
    if (!done)
      finish();
  }

  void finish() {
    using namespace std::chrono;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start_).count();
    std::cerr << name_ << "\t" << duration << " μs\n";
    done = true;
  }

  bool done;

  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};
