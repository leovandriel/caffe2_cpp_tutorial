#include <caffe2/core/context_gpu.h>
#include "caffe2/operator/diagonal_op.h"

namespace caffe2 {

int diagonal_op_step(const Tensor<CUDAContext>& tensor) {
  auto step = 0;
  for (auto d : tensor.dims()) {
    step = step * d + 1;
  }
  return step;
}

int diagonal_op_size(const Tensor<CUDAContext>& tensor) {
  auto size = tensor.dim(0);
  for (auto d : tensor.dims()) {
    if (size > d) size = d;
  }
  return size;
}

int diagonal_op_offset(const Tensor<CUDAContext>& tensor,
                       const std::vector<TIndex>& offset) {
  auto off = 0, i = 0;
  for (auto d : tensor.dims()) {
    off = off * d + offset[i++];
  }
  return off;
}

namespace {

__global__ void DiagonalKernel(const int N, const int C, const int D,
                               const float* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) { Y[i] = X[i * C + D]; }
}

}  // namespace

template <>
bool DiagonalOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto size = diagonal_op_size(X);
  Y->Resize(size);
  if (size > 0) {
    auto step = diagonal_op_step(X);
    auto offset = diagonal_op_offset(X, offset_);
    DiagonalKernel<<<CAFFE_GET_BLOCKS(Y->size()), CAFFE_CUDA_NUM_THREADS, 0,
                     context_.cuda_stream()>>>(
        Y->size(), step, offset, X.data<float>(), Y->mutable_data<float>());
  }
  return true;
}

namespace {

__global__ void DiagonalGradientKernel(const int N, const int C, const int D,
                                       const float* dY, float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = (i >= D && (i - D) % C == 0 ? dY[i] : 0);
  }
}

}  // namespace

template <>
bool DiagonalGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  auto size = diagonal_op_size(X);
  DCHECK_EQ(dY.size(), size);
  if (size > 0) {
    auto step = diagonal_op_step(X);
    auto offset = diagonal_op_offset(X, offset_);
    DiagonalGradientKernel<<<CAFFE_GET_BLOCKS(dX->size()),
                             CAFFE_CUDA_NUM_THREADS, 0,
                             context_.cuda_stream()>>>(
        dX->size(), step, offset, dY.data<float>(), dX->mutable_data<float>());
  }
  return true;
}

REGISTER_CUDA_OPERATOR(Diagonal, DiagonalOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(DiagonalGradient,
                       DiagonalGradientOp<float, CUDAContext>);

}  // namespace caffe2
