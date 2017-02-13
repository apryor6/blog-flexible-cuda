all: cpu
clean:
	-rm demo
cpu:
	g++ -std=c++11 main.cpp -o demo
gpu:
	nvcc -std=c++11 main.cpp -o demo -DENABLE_GPU ArrayPow2_CUDA.cu
