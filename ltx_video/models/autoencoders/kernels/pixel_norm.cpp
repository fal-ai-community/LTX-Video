#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <vector>

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
);

void pixel_norm_inplace(at::Tensor &x, at::Tensor &scale, at::Tensor &shift, float eps) {
    int num_pixels = x.size(2)*x.size(3)*x.size(4);
    int batch_size = x.size(0);
    int num_channels = x.size(1);
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    pixel_norm_cuda<at::BFloat16>(
        x.data_ptr(),
        scale.data_ptr(),
        shift.data_ptr(),
        x.data_ptr(),
        eps,
        batch_size,
        num_channels,
        num_pixels,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pixel_norm_inplace", &pixel_norm_inplace,
          "pixel norm inplace");
}
