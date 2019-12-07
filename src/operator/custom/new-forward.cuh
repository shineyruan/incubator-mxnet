#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

/**
 * Enable __CUDACC__ in header files                   
 * Although it is defined elsewhere, we need it to get 
 * better linting.
 */
#ifndef __CUDACC__
#define __CUDACC__

#include <mxnet/base.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#endif

namespace mxnet {
namespace op {

#define TILE_SIZE 8

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
    /*
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
#define y4d(b, m, h, w) y[(b) * (M * H_out * W_out) + (m) * (H_out * W_out) + (h) * (W_out) + w]
#define x4d(b, c, h, w) x[(b) * (C * H * W) + (c) * (H * W) + (h) * (W) + w]
#define k4d(m, c, h, w) k[(m) * (C * K * K) + (c) * (K * K) + (h) * (K) + w]

    /*****************************************************************/
    /* OPTIMIZATION CODE STARTS HERE */

    extern __shared__ float s[];
    float *kernel = s;             // shared memory for kernel
    float *tile = &kernel[K * K];  // shared memory for tile

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int global_m = blockIdx.z * blockDim.z + tz;

    int x_o = blockIdx.x * TILE_SIZE + tx;
    int y_o = blockIdx.y * TILE_SIZE + ty;
    int kernelRadius = K / 2;
    int tileW = blockDim.x;

    int x_i = x_o;
    int y_i = y_o;

    for (int b = 0; b < B; ++b) {
        float result = 0.0f;
        for (int c = 0; c < C; ++c) {
            float perChannelSum = 0.0f;

            /* load tile from input */
            if (y_i >= 0 && y_i < H && x_i >= 0 && x_i < W) {
                tile[ty * tileW + tx] = x4d(b, c, y_i, x_i);
            } else {
                tile[ty * tileW + tx] = 0.0f;
            }
            __syncthreads();

            /* load kernel from global memory */
            if (tx < K && ty < K && global_m < M) {
                kernel[ty * K + tx] = k4d(global_m, c, ty, tx);
            }
            __syncthreads();

            /* 2-D convolution */
            if (tx < TILE_SIZE && ty < TILE_SIZE) {
                for (int i = 0; i < kernelRadius; ++i) {
                    for (int j = 0; j < kernelRadius; ++j) {
                        perChannelSum += kernel[i * K + j] * tile[(ty + i) * tileW + j];
                    }
                }
            }

            result += perChannelSum;
        }

        if (y_o < H_out && x_o < W_out && global_m < M) {
            y4d(b, global_m, y_o, x_o) = result;
        }
    }

    /*****************************************************************/
    /* DEPRECATED ORIGINAL CODE FOR REFERENCE */

    // int b = blockDim.x * blockIdx.x + threadIdx.x;

    // if (b < B)  // for each image in the batch
    // {
    //     for (int m = 0; m < M; m++)          // for each output feature maps
    //         for (int h = 0; h < H_out; h++)  // for each output element
    //             for (int w = 0; w < W_out; w++) {
    //                 y4d(b, m, h, w) = 0;
    //                 for (int c = 0; c < C; c++)      // sum over all input feature maps
    //                     for (int p = 0; p < K; p++)  // KxK filter
    //                         for (int q = 0; q < K; q++)
    //                             y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
    //             }
    // }

    /*****************************************************************/

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   We only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    /**
     * Some information about the GPU device.
     *  Device [0] name: TITAN Xp
     *  Maximum thread per block: 1024
     *  Max block dimensions: 1024x1024x64
     */

    // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1)

    const int B = x.shape_[0];
    const int M = y.shape_[1];  // num_filter
    const int yH = y.shape_[2];
    const int yW = y.shape_[3];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    /* ORIGINAL DEFINITION */
    // dim3 gridDim((B + 511) / 512);
    // dim3 blockDim(512);
    /* ******************* */
    printf("Current kernel size: %d\n", K);

    int kernelRadius = K / 2;
    int blockSize = TILE_SIZE + kernelRadius * 2;
    dim3 gridDim((int)ceil(yH * 1.0 / TILE_SIZE), (int)ceil(yW * 1.0 / TILE_SIZE), (int)ceil(M * 1.0 / TILE_SIZE));
    dim3 blockDim(blockSize, blockSize, TILE_SIZE);

    /* for dynamic shared memory allocation */
    /* Reference: https://devblogs.nvidia.com/using-shared-memory-cuda-cc/ */
    int kernelLength = K * K;
    int tileLength = blockSize * blockSize;  // a 2-D convolution tile total length

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    forward_kernel<<<gridDim, blockDim,
                     (kernelLength + tileLength) * sizeof(float)>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

#undef TILE_SIZE

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert(0 && "No forward implementation for other datatypes needed");
}
}  // namespace op
}  // namespace mxnet

#endif