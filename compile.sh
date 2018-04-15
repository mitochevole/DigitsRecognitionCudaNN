mkdir -p PROVA/build/Debug/CUDA2-Linux-x86/_ext/2078672788
nvcc -std=c++11 --generate-code arch=compute_20,code=sm_21 -rdc=true   -c -g -I/usr/include -I/usr/local/cuda/include -o PROVA/build/Debug/CUDA2-Linux-x86/_ext/2078672788/main.o /home/michele/Dropbox/PhD/NetBeansProjects/Neural_Network/main.cu

mkdir -p PROVA/dist/Debug/CUDA2-Linux-x86
nvcc -std=c++11 --generate-code arch=compute_20,code=sm_21 -rdc=true    -o PROVA/dist/Debug/CUDA2-Linux-x86/neural_network build/Debug/CUDA2-Linux-x86/_ext/2078672788/main.o -L/usr/local/cuda-8.0/lib64 -lcurand -lcublas


