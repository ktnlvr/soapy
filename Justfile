run:
    python3 reference_implementation.py
    mkdir -p build
    clang++ main.cpp -I. -Iinclude/ -lvulkan cnpy.cpp  -lz -o build/main
    ./build/main

alias cs := compile-shaders

compile-shaders:
    glslangValidator -V xi_lmk.comp -o xi_lmk.spv
