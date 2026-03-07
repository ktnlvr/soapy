build:
    mkdir -p build/
    nvcc main.cu -o build/main -lm -lgsl -lgslcblas 
    time ./build/main
