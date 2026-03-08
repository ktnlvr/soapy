build:
    mkdir -p build/
    nvcc main.cu -o build/main -lm -lgsl -lgslcblas -O3
    time ./build/main
