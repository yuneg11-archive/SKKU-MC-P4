CUDA=/usr/local/cuda/
CC=nvcc

all: counting_sort.cu driver.cpp
	$(CC) -o driver counting_sort.cu driver.cpp

clean:
	rm -rf driver
