default:
    just --list

run-py:
    python3 reference_implementation.py

comp-cpp:
    mkdir -p build
    clang++ main.cpp -I. -Iinclude/ -lvulkan cnpy.cpp  -lz -o build/main -std=gnu++23
    
run-cpp:
    ./build/main

alias cs := compile-shaders

compile-shaders:
    glslangValidator -V xi_lmk.comp -o xi_lmk.spv
    glslangValidator -V c_nlm.comp -o c_nlm.spv

alias b := bench

bench:
    python3 bench.py
