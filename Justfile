build:
    mkdir -p build/
    clang main.c -o build/main -lm -lgsl -lgslcblas 
    time ./build/main
