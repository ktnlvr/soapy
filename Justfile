run:
    python3 reference_implementation.py
    rm -rf build
    mkdir -p build
    cmake -B build
    cmake --build build

alias cs := compile-shaders

compile-shaders:
    glslangValidator -V xi_lmk.comp -o xi_lmk.spv
