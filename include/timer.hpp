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

  long long finish() {
    using namespace std::chrono;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start_).count();
    std::cout << name_ << "\t" << duration << " μs\n";
    done = true;
    return duration;
  }

  bool done;

  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};
