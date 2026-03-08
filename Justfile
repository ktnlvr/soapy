build:
    mkdir -p build/
    nvcc main.cu -g -lineinfo  -o build/main -lm -lgsl -lgslcblas -O3
    time ./build/main

dbg:
    mkdir -p build/
    nvcc main.cu -g -lineinfo -Xcompiler -fno-omit-frame-pointer -o build/maindbg -lm -lgsl -lgslcblas

profile-dbg: dbg
    nsys profile -o report --sample=cpu --trace=cuda,osrt --backtrace=fp build/maindbg

warmup:
    python3 generate.py
    vmtouch -t random_hydrogens.xyz
