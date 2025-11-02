run:
    python3 reference_implementation.py
    mkdir -p build
    clang++ main.cpp -I. -Iinclude/ -lvulkan cnpy.cpp  -lz -o build/main -std=gnu++23
    ./build/main

alias cs := compile-shaders

compile-shaders:
    glslangValidator -V xi_lmk.comp -o xi_lmk.spv
    glslangValidator -V c_nlm.comp -o c_nlm.spv

alias b := bench

bench:
    python3 bench.py
