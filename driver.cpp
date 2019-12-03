#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <cuda_runtime.h>

#define N INT_MAX // 8 GB (= 2,147,483,648 * 4 Byte)
#define MAX_VAL INT_MAX // 8 GB (= 2,147,483,648 * 4 Byte)

extern void counting_sort(int arr[], int size, int max_val);
void host_counting_sort(int arr[], int size, int max_val) {
    int *histogram = new int[max_val]();

    for (int i = 0; i < size; i++) {
        histogram[arr[i]]++;
    }

    for (int idx = 0, i = 0; i < max_val; i++) {
        for (int j = 0; j < histogram[i]; j++) {
            arr[idx++] = i;
        }
    }

    delete [] histogram;
}

bool is_sorted(int arr[], int size) {
    for(int i = 0; i < size - 1; i++){
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

bool is_equal(int arr1[], int arr2[], int size) {
    for (int i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

double get_elapse_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + ((double) (end.tv_nsec - start.tv_nsec)) / 1000000000;
}

int main(int argc, char *argv[]) {
    bool cuda = true;
    bool verify = false;

    if (argc == 2) {
        if (strcmp(argv[1], "host") == 0) {
            cuda = false;
        } else if (strcmp(argv[1], "verify") == 0) {
            verify = true;
        }
    }

    int *arr_device = (cuda ? new int[N] : NULL);
    int *arr_host = (cuda == verify ? new int[N] : NULL);
    struct timespec time1, time2, time3;

    if (cuda && !verify) {
        printf("Device only\n");

        for(int i = 0; i < N; i++){
            arr_device[i] = rand() % MAX_VAL;
        }

        clock_gettime(CLOCK_REALTIME, &time1);
        counting_sort(arr_device, N, MAX_VAL);
        clock_gettime(CLOCK_REALTIME, &time2);

        printf("Device sorting time: %.3lf secs\n", get_elapse_time(time1, time2));
        printf("%s\n", (is_sorted(arr_device, N) ? "Sorted" : "Not sorted"));
    } else if (!cuda && !verify) {
        printf("Host only\n");

        for(int i = 0; i < N; i++){
            arr_host[i] = rand() % MAX_VAL;
        }

        clock_gettime(CLOCK_REALTIME, &time1);
        host_counting_sort(arr_host, N, MAX_VAL);
        clock_gettime(CLOCK_REALTIME, &time2);

        printf("Host sorting time: %.3lf secs\n", get_elapse_time(time1, time2));
        printf("%s\n", (is_sorted(arr_host, N) ? "Sorted" : "Not sorted"));
    } else if (cuda && verify) {
        printf("Device and Host\n");

        for(int i = 0; i < N; i++){
            arr_device[i] = arr_host[i] = rand() % MAX_VAL;
        }

        clock_gettime(CLOCK_REALTIME, &time1);
        counting_sort(arr_device, N, MAX_VAL);
        clock_gettime(CLOCK_REALTIME, &time2);
        host_counting_sort(arr_host, N, MAX_VAL);
        clock_gettime(CLOCK_REALTIME, &time3);

        printf("Device sorting time: %.3lf secs\n", get_elapse_time(time1, time2));
        printf("Host sorting time: %.3lf secs\n", get_elapse_time(time2, time3));
        printf("%s\n", (is_equal(arr_device, arr_host, N) ? "Sorted (verified)" : "Not sorted"));
    }

    return 0;
}
