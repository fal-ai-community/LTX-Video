#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <limits>
#include <cuda.h>

#include <ATen/ATen.h>
#include <torch/types.h>

using bfloat16 = at::BFloat16;
using accessor = at::PackedTensorAccessor64<at::BFloat16, 5>;
constexpr int BLOCK_ROWS = 8;

template <typename accessor_t>
__global__ void add_tensors_tiled(const accessor_t tensor1,
                                  const accessor_t tensor2,
                                  accessor_t result,
                                  int offset
) {
  int B = tensor1.size(0);
  int C = tensor1.size(1);
  int T = tensor1.size(2);
  int H = tensor1.size(3);
  int W = tensor1.size(4);

  const int outer_size = B * C;
  const int inner_dim2 = H * W;
  const int hw_ind = blockIdx.z * blockDim.x + threadIdx.x;
  const int h = hw_ind / W;
  const int w = hw_ind % W;
  if (h >= H || w >= W) {
    return;
  }
  const int t = blockIdx.y;
  const int block_bc = blockIdx.x * BLOCK_ROWS;

  const int b = block_bc / C;
  const int c = block_bc % C;

  for (int i = 0; i < BLOCK_ROWS; i++) {
    result[b][c + i][t][h][w] = tensor1[b][c + i][t][h][w] + tensor2[b][c + i][t + offset][h][w];
  }
}

void inplace_add_cuda(
  at::Tensor& x,
  at::Tensor& workspace,
  int offset,
  cudaStream_t stream
) {

  int B = x.size(0);
  int C = x.size(1);
  int T = x.size(2);
  int H = x.size(3);
  int W = x.size(4);

  int outer_size = B * (C / BLOCK_ROWS);
  int inner_dim2 = H * W;
  const int max_threads_per_block = 1024;
  int threads_per_block_x = (inner_dim2 > max_threads_per_block) ? max_threads_per_block : inner_dim2;
  int threads_per_block_y = (T > max_threads_per_block / threads_per_block_x) ? max_threads_per_block / threads_per_block_x : T;

  dim3 threadsPerBlock(threads_per_block_x, threads_per_block_y);
  int max_v = (threadsPerBlock.x * threadsPerBlock.y);
  int inner_dim = ((inner_dim2 + max_v - 1) / max_v);

  dim3 blocksPerGrid(outer_size, T, inner_dim);
  if (x.numel() >= std::numeric_limits<int>::max() ||
  workspace.numel() >= std::numeric_limits<int>::max()) {
    add_tensors_tiled<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      x.packed_accessor64<at::BFloat16, 5, torch::RestrictPtrTraits>(),
      workspace.packed_accessor64<at::BFloat16, 5, torch::RestrictPtrTraits>(),
      x.packed_accessor64<at::BFloat16, 5, torch::RestrictPtrTraits>(),
      offset
    );
  } else {
    add_tensors_tiled<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      x.packed_accessor32<at::BFloat16, 5, torch::RestrictPtrTraits>(),
      workspace.packed_accessor32<at::BFloat16, 5, torch::RestrictPtrTraits>(),
      x.packed_accessor32<at::BFloat16, 5, torch::RestrictPtrTraits>(),
      offset
    );
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void inplace_add_cuda(at::Tensor&, at::Tensor&, int, cudaStream_t);
