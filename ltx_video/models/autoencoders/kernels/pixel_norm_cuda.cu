
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>

template <int CHANNELS_, int NUM_WARPS_, typename input_t_>
struct pixel_norm_kernel_traits {
    static constexpr int CHANNELS = CHANNELS_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    using input_t = input_t_;
    using copy_t = uint4;

    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int NUM_ELEMS_PER_COPY = sizeof(copy_t) / sizeof(input_t);
    static constexpr int NUM_THREADS_PER_PIXEL = CHANNELS / NUM_ELEMS_PER_COPY;
    static constexpr int NUM_WARPS_PER_PIXEL = NUM_THREADS_PER_PIXEL / 32;
    static constexpr int PIXELS_PER_BLOCK = NUM_THREADS / NUM_THREADS_PER_PIXEL;
};

template <int NUM_ELEMS, typename input_t, typename copy_t, int COPY_THREADS>
inline __device__ void load_input(const input_t *x, float x_vals[NUM_ELEMS]) {;
    input_t x_vals_load[NUM_ELEMS] = {0};
    reinterpret_cast<copy_t*>(x_vals_load)[0] = reinterpret_cast<const copy_t*>(x)[threadIdx.x % COPY_THREADS];
    for (int i = 0; i < NUM_ELEMS; i++) {
        x_vals[i] = float(x_vals_load[i]);
    }
}

inline __device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

template <typename kernel_traits>
__global__ void pixel_norm_kernel(
    const typename kernel_traits::input_t* input,
    const typename kernel_traits::input_t* scale,
    const typename kernel_traits::input_t* shift,
    typename kernel_traits::input_t* output,
    const float eps,
    const int num_pixels,
    const int batch_size
) {
    extern __shared__ float smem_[];

    const int tid = threadIdx.x;
    const int pixel_id = blockIdx.x * kernel_traits::PIXELS_PER_BLOCK;
    const int batch_id = pixel_id / num_pixels;
    const int total_pixels = num_pixels * batch_size;
    const int processing_pixels = pixel_id + tid / kernel_traits::NUM_THREADS_PER_PIXEL;

    if (processing_pixels >= total_pixels) {
        return;
    }

    float x_vals[kernel_traits::NUM_ELEMS_PER_COPY];
    float sum = 0.0f;

    const typename kernel_traits::input_t* x_ptr = input + static_cast<long long>(pixel_id) * static_cast<long long>(kernel_traits::CHANNELS);
    typename kernel_traits::input_t* output_ptr = output + static_cast<long long>(pixel_id) * static_cast<long long>(kernel_traits::CHANNELS);

    const typename kernel_traits::input_t* scale_ptr = scale + batch_id * kernel_traits::CHANNELS;
    const typename kernel_traits::input_t* shift_ptr = shift + batch_id * kernel_traits::CHANNELS;

    float scale_vals[kernel_traits::NUM_ELEMS_PER_COPY];
    float shift_vals[kernel_traits::NUM_ELEMS_PER_COPY];

    load_input<kernel_traits::NUM_ELEMS_PER_COPY, typename kernel_traits::input_t, typename kernel_traits::copy_t, kernel_traits::NUM_THREADS>(x_ptr, x_vals);
    load_input<kernel_traits::NUM_ELEMS_PER_COPY, typename kernel_traits::input_t, typename kernel_traits::copy_t, kernel_traits::NUM_THREADS_PER_PIXEL>(scale_ptr, scale_vals);
    load_input<kernel_traits::NUM_ELEMS_PER_COPY, typename kernel_traits::input_t, typename kernel_traits::copy_t, kernel_traits::NUM_THREADS_PER_PIXEL>(shift_ptr, shift_vals);

    for (int i = 0; i < kernel_traits::NUM_ELEMS_PER_COPY; i++) {
        sum += x_vals[i] * x_vals[i];
    }

    for (int i = 1; i < kernel_traits::NUM_THREADS_PER_PIXEL; i *= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, i);
    }
    if constexpr (kernel_traits::NUM_WARPS_PER_PIXEL > 1) {

        smem_[tid / 32] = sum;
        __syncthreads();
        sum = 0.0f;
        for (int i = 0; i < kernel_traits::NUM_WARPS_PER_PIXEL; i++) {
            sum += smem_[i];
        }
    }

    sum = sum / kernel_traits::CHANNELS;

    float inv_sqrt_sum = rsqrtf(sum + eps);
    #pragma unroll
    for (int i = 0; i < kernel_traits::NUM_ELEMS_PER_COPY; i++) {
        x_vals[i] *= inv_sqrt_sum;
        x_vals[i] = (1.0f + scale_vals[i]) * x_vals[i] + shift_vals[i];
        x_vals[i] = silu(x_vals[i]);
    }
    typename kernel_traits::input_t out_vals_store[kernel_traits::NUM_ELEMS_PER_COPY];
    for (int i = 0; i < kernel_traits::NUM_ELEMS_PER_COPY; i++) {
        out_vals_store[i] = static_cast<typename kernel_traits::input_t>(x_vals[i]);
    }
    reinterpret_cast<typename kernel_traits::copy_t*>(output_ptr)[tid] = reinterpret_cast<const typename kernel_traits::copy_t*>(out_vals_store)[0];
}

template <int NUM_WARPS, int CHANNELS, typename input_t, typename output_t>
void pixel_norm_launch(
    const void* input,
    const void* scale,
    const void* shift,
    void* output,
    const float eps,
    const int batch_size,
    const int num_pixels,
    cudaStream_t stream
) {

    using kernel_traits = pixel_norm_kernel_traits<CHANNELS, NUM_WARPS, input_t>;
    int BLOCK_SIZE = (num_pixels*batch_size + kernel_traits::PIXELS_PER_BLOCK - 1) / kernel_traits::PIXELS_PER_BLOCK;

    dim3 block(kernel_traits::NUM_THREADS);
    dim3 grid(BLOCK_SIZE);

    auto kernel = &pixel_norm_kernel<kernel_traits>;
    kernel<<<grid, block, 8192, stream>>>(
        (const typename kernel_traits::input_t*)input,
        (const typename kernel_traits::input_t*)scale,
        (const typename kernel_traits::input_t*)shift,
        (typename kernel_traits::input_t*)output,
        eps,
        num_pixels,
        batch_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t>
void pixel_norm_cuda(
    const void* input,
    const void* scale,
    const void* shift,
    void* output,
    const float eps,
    const int batch_size,
    const int num_channels,
    const int num_pixels,
    cudaStream_t stream
) {
    if (num_channels == 128) {
        pixel_norm_launch<2, 128, input_t, input_t>(input, scale, shift, output, eps, batch_size, num_pixels, stream);
    } else if (num_channels == 256) {
        pixel_norm_launch<4, 256, input_t, input_t>(input, scale, shift, output, eps, batch_size, num_pixels, stream);
    } else if (num_channels == 512) {
        pixel_norm_launch<4, 512, input_t, input_t>(input, scale, shift, output, eps, batch_size, num_pixels, stream);
    } else if (num_channels == 1024) {
        pixel_norm_launch<8, 1024, input_t, input_t>(input, scale, shift, output, eps, batch_size, num_pixels, stream);
    }
}

template void pixel_norm_cuda<at::BFloat16>(const void*, const void*, const void*, void*, float, int, int, int, cudaStream_t);
