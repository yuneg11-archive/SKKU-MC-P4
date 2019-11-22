#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1000000
#define MAX_VAL 1000000

extern void counting_sort(int arr[], int size, int max_val);

int main() {
    int array[N];

    for(int i = 0; i < N; i++){
        array[i] = rand() % MAX_VAL;
    }

    counting_sort(array, N, MAX_VAL);

    for(int i = 0; i < N - 1; i++){
        if (array[i] > array[i + 1]) {
            printf("Not sorted\n");
            return -1;
        }
    }
    printf("Sorted\n");

    return 0;
}
