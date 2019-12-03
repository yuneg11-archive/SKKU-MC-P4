#include <cuda.h>

#define THREAD_NUM 1024

// Tuned for NVIDIA Tesla V100 (12 GB VRAM)
#define ARR_SPLIT_LEN 805306368 // 3 GB (= 805,306,368 * 4 Byte)

// Util

__host__ int array_split(size_t arr_len[], int size) {
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
    return arr_len_cnt;
}

// Histogram

__global__ void build_histogram_kernel(int arr[], int histogram[], int arr_size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < arr_size) {
        atomicAdd(&histogram[arr[idx]], 1);
    }
}

__host__ void build_histogram(int arr[], int histogram[], int size, int max_val) {
    int *arr_device;
    size_t arr_len[3] = { 0, 0, 0 };
    int arr_len_cnt = array_split(arr_len, size);

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
    cudaFree(arr_device);
    cudaFree(histogram_device);
}

// Prefix

__host__ void build_prefix(int histogram[], int max_val) {
    int sum = 0;
    for (int i = 0; i < max_val; i++) {
        sum += histogram[i];
        histogram[i] = sum;
    }
}

// Output

__global__ void build_output_kernel(int arr[], int prefix[], int prefix_size, int first_prefix, int base_idx) {
    __shared__ int local_prefix[THREAD_NUM + 1];
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x + 1;

    if (global_idx == 0) {
        local_prefix[0] = first_prefix;
    } else if (local_idx == 1) {
        local_prefix[0] = prefix[global_idx - 1];
    }

    if (global_idx < prefix_size) {
        local_prefix[local_idx] = prefix[global_idx];

        __syncthreads();

        int cnt = local_prefix[local_idx] - local_prefix[local_idx - 1];
        int start_idx = local_prefix[local_idx - 1];

        for (int i = 0; i < cnt; i++) {
            arr[start_idx + i] = global_idx + base_idx;
        }
    }
}

__host__ void build_output(int arr_out[], int prefix[], int size, int max_val) {
    int *prefix_device;
    size_t prefix_len[3] = { 0, 0, 0 };
    int prefix_len_cnt = array_split(prefix_len, max_val);

    int *arr_out_device;
    int arr_out_len = size;

    cudaMalloc(&arr_out_device, arr_out_len * sizeof(int));
    cudaMalloc(&prefix_device, prefix_len[0] * sizeof(int));

    for (int i = 0; i < prefix_len_cnt; i++) {
        int block_num = (prefix_len[i] / THREAD_NUM) + (prefix_len[i] % THREAD_NUM == 0 ? 0 : 1);
        int first_prefix = (i == 0 ? 0 : prefix[i * ARR_SPLIT_LEN - 1]);
        cudaMemcpy(prefix_device, &prefix[i * ARR_SPLIT_LEN], prefix_len[i] * sizeof(int), cudaMemcpyHostToDevice);
        build_output_kernel <<< block_num, THREAD_NUM >>> (arr_out_device, prefix_device, prefix_len[i], first_prefix, i * ARR_SPLIT_LEN);
    }

    cudaMemcpy(arr_out, arr_out_device, arr_out_len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(arr_out_device);
    cudaFree(prefix_device);
}

// Sort

__host__ void counting_sort(int arr[], int size, int max_val) {
    int *histogram_and_prefix = new int[max_val];
    build_histogram(arr, histogram_and_prefix, size, max_val);
    build_prefix(histogram_and_prefix, max_val);
    build_output(arr, histogram_and_prefix, size, max_val);
    delete [] histogram_and_prefix;
}
