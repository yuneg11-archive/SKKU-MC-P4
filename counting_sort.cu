#include <cuda.h>

#define THREAD_NUM 1024

// Tuned for NVIDIA Tesla V100 (12 GB VRAM)
#define ARR_SPLIT_LEN 805306368 // 3 GB (= 805,306,368 * 4 Byte)

__global__ void build_histogram_kernel(int arr[], int histogram[], int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&histogram[arr[idx]], 1);
    }
}

__host__ void counting_sort(int arr[], int size, int max_val) {
    int *arr_device;
    size_t arr_len[3] = { 0, 0, 0 };
    int arr_len_cnt = 0;
    if (size > 2 * ARR_SPLIT_LEN) {
        arr_len_cnt = 3;
        arr_len[0] = arr_len[1] = ARR_SPLIT_LEN;
        arr_len[2] = size - 2 * ARR_SPLIT_LEN;
    } else if (size > ARR_SPLIT_LEN) {
        arr_len_cnt = 2;
        arr_len[0] = ARR_SPLIT_LEN;
        arr_len[1] = size - ARR_SPLIT_LEN;
    } else {
        arr_len_cnt = 1;
        arr_len[0] = size;
    }

    int *histogram = new int[max_val];
    int *histogram_device;
    size_t histogram_len = max_val;


    cudaMalloc(&arr_device, arr_len[0] * sizeof(int));
    cudaMalloc(&histogram_device, histogram_len * sizeof(int));
    cudaMemset(histogram_device, 0, histogram_len * sizeof(int));

    for (int i = 0; i < arr_len_cnt; i++) {
        int block_num = (arr_len[i] / THREAD_NUM) + (arr_len[i] % THREAD_NUM == 0 ? 0 : 1);
        cudaMemcpy(arr_device, &arr[i * ARR_SPLIT_LEN], arr_len[i] * sizeof(int), cudaMemcpyHostToDevice);
        build_histogram_kernel <<< block_num, THREAD_NUM >>> (arr_device, histogram_device, arr_len[i]);
    }

    cudaMemcpy(histogram, histogram_device, histogram_len * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < histogram_len; i++) {
        sum += histogram[i];
        histogram[i] = sum;
    }

    for (int j = 0; j < histogram[0]; j++) {
        arr[j] = 0;
    }
    for (int i = 1; i < histogram_len; i++) {
        int cnt = histogram[i] - histogram[i - 1];
        int base_idx = histogram[i - 1];
        for (int j = 0; j < cnt; j++) {
            arr[base_idx + j] = i;
        }
    }

    cudaFree(arr_device);
    cudaFree(histogram_device);
    delete [] histogram;
}
