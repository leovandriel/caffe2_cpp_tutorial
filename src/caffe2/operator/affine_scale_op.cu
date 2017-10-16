#include <caffe2/core/context_gpu.h>
#include "caffe2/operator/affine_scale_op.h"

namespace caffe2 {

namespace {

__global__ void AffineScaleKernel(const int N, const int C, const float* X,
                                  const float* M, const float* S, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) { Y[i] = X[i] * S[i / C] + M[i / C]; }
}

__global__ void AffineScaleInverseKernel(const int N, const int C,
                                         const float* X, const float* M,
                                         const float* S, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) { Y[i] = (X[i] - M[i / C]) / (S[i / C] + 1e-8); }
}

}  // namespace

template <>
bool AffineScaleOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& M = Input(1);
  auto& S = Input(2);
  auto* Y = Output(0);
  DCHECK_EQ(M.size(), X.dim(0));
  DCHECK_EQ(S.size(), X.dim(0));
  Y->ResizeLike(X);
  if (X.size() > 0) {
    auto size = X.size() / X.dim(0);
    if (inverse_) {
      AffineScaleInverseKernel<<<CAFFE_GET_BLOCKS(X.size()),
                                 CAFFE_CUDA_NUM_THREADS, 0,
                                 context_.cuda_stream()>>>(
          X.size(), size, X.data<float>(), M.data<float>(), S.data<float>(),
          Y->mutable_data<float>());
    } else {
      AffineScaleKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS, 0,
                          context_.cuda_stream()>>>(
          X.size(), size, X.data<float>(), M.data<float>(), S.data<float>(),
          Y->mutable_data<float>());
    }
  }
  return true;
}

namespace {

__global__ void AffineScaleGradientKernel(const int N, const int C,
                                          const float* dY, const float* S,
                                          float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) { dX[i] = dY[i] * S[i / C]; }
}

__global__ void AffineScaleInverseGradientKernel(const int N, const int C,
                                                 const float* dY,
                                                 const float* S, float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) { dX[i] = dY[i] / (S[i / C] + 1e-8); }
}

}  // namespace

template <>
bool AffineScaleGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& S = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(S.size(), X.dim(0));
  DCHECK_EQ(dY.size(), X.size());
  dX->ResizeLike(X);
  if (X.size() > 0) {
    auto size = X.size() / X.dim(0);
    if (inverse_) {
      AffineScaleInverseGradientKernel<<<CAFFE_GET_BLOCKS(dY.size()),
                                         CAFFE_CUDA_NUM_THREADS, 0,
                                         context_.cuda_stream()>>>(
          dY.size(), size, dY.data<float>(), S.data<float>(),
          dX->mutable_data<float>());
    } else {
      AffineScaleGradientKernel<<<CAFFE_GET_BLOCKS(dY.size()),
                                  CAFFE_CUDA_NUM_THREADS, 0,
                                  context_.cuda_stream()>>>(
          dY.size(), size, dY.data<float>(), S.data<float>(),
          dX->mutable_data<float>());
    }
  }
  return true;
}

REGISTER_CUDA_OPERATOR(AffineScale, AffineScaleOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(AffineScaleGradient,
                       AffineScaleGradientOp<float, CUDAContext>);

}  // namespace caffe2
