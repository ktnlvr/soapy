default:
    just --list

clean:
    rm -rf build
    rm -rf *.npy

run-py:
    python3 reference_implementation.py

comp-sycl:
    mkdir -p build
    icpx -fsycl sycl.cpp -o build/sycl -Iinclude cnpy.cpp -lz -g

comp-cpp:
    mkdir -p build
    clang++ main.cpp -I. -Iinclude/ -lvulkan cnpy.cpp -lz -o build/main -std=gnu++23

run-cpp:
    ./build/main

run-sycl:
    ./build/sycl

sycl:
    just clean && just run-py && just comp-sycl && just run-sycl && just bench

alias cs := compile-shaders

compile-shaders:
    glslangValidator -V xi_lmk.comp -o xi_lmk.spv
    glslangValidator -V c_nlm.comp -o c_nlm.spv

alias b := bench

bench:
    python3 bench.py
