#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <climits>

#include "bitonic_sort.h"

static int THREADS_IN_BLOCK = 1024;

using namespace std;

// void handleRes(CUresult res, string message) {
//     if (res != CUDA_SUCCESS) {
//         printf("message", res);
//         exit(1);
//     }
// }

int* bitonic_sort(int* to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS) {
        printf("cannot acquire device 0\n");
        exit(1);
    }
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS) {
        printf("cannot create Kontext\n");
        exit(1);
    }

    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "bitonic_sort.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);
        exit(1);
    }

    CUfunction odd_even_phase1;
    res = cuModuleGetFunction(&odd_even_phase1, cuModule, "odd_even_phase1");
    if (res != CUDA_SUCCESS) {
        printf("some error %d\n", __LINE__);
        exit(1);
    }
    CUfunction odd_even_phase2;
    res = cuModuleGetFunction(&odd_even_phase2, cuModule, "odd_even_phase2");
    if (res != CUDA_SUCCESS) {
        printf("some error %d\n", __LINE__);
        exit(1);
    }


    int numberOfBlocks = (size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    int max_grid_dim_x = 32768;
    int x_dim = numberOfBlocks > max_grid_dim_x ? max_grid_dim_x : numberOfBlocks;
    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;


    int* result = (int*) malloc(sizeof(int) * size);
    cuMemHostRegister((void*) result, size * sizeof(int), 0);
    cuMemHostRegister((void*) to_sort, size * sizeof(int), 0);

    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size * sizeof(int));
    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));

    // void* args[2] =  { &deviceToSort, &size};
    // cuLaunchKernel(bitonic_sort, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args, 0);
    // cuCtxSynchronize();


    int n;
    //fit n to power of 2
    for (n = 1; n < size; n <<= 1);

    for (int batch_size = 1; batch_size <= n; batch_size *= 2) {
        void* args1[3] = { &deviceToSort, &batch_size, &size};
        res = cuLaunchKernel(odd_even_phase1, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0);
        if (res != CUDA_SUCCESS) {
            printf("some error %d\n", __LINE__);
            exit(1);
        }
        for (int d = batch_size / 2; d >= 1; d *= 2) {
            void* args2[3] = { &deviceToSort, &d, &batch_size, &size};

            res = cuLaunchKernel(odd_even_phase2, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args2, 0);
            if (res != CUDA_SUCCESS) {
                printf("some error %d\n", __LINE__);
                exit(1);
            }
            cuCtxSynchronize();
        }

    }
    cuCtxSynchronize();

    cuMemcpyDtoH((void*)result, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(result);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
    return result;
}

