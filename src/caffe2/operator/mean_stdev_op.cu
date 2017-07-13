#include "caffe2/operator/mean_stdev_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

__global__ void ZeroKernel(const int N, float* X) {
  CUDA_1D_KERNEL_LOOP(i, N) {
   X[i] = 0;
  }
}

__global__ void SumKernel(const int N, const int C, const float* X, float* M) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    M[i / C] += X[i];
  }
}

__global__ void ScaleKernel(const int N, const float C, float* M) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    M[i] /= C;
  }
}

__global__ void SumSqKernel(const int N, const int C, const float* X, const float* M, float* S) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float d = X[i] - M[i / C];
    S[i / C] += d * d;
  }
}

__global__ void SqrtKernel(const int N, const float C, float* S) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    S[i] = sqrtf(S[i] / C);
  }
}

}  // namespace

template <>
bool MeanStdevOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* M = Output(0);
  auto* S = Output(1);
  M->Resize(X.dim(0));
  S->Resize(X.dim(0));
  if (X.size() > 0) {
    auto size = X.size() / X.dim(0);

    ZeroKernel<<<CAFFE_GET_BLOCKS(M->size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      M->size(), M->mutable_data<float>());
    ZeroKernel<<<CAFFE_GET_BLOCKS(S->size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      S->size(), S->mutable_data<float>());

    SumKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      X.size(), size, X.data<float>(), M->mutable_data<float>());
    ScaleKernel<<<CAFFE_GET_BLOCKS(M->size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      M->size(), (float)size, M->mutable_data<float>());

    SumSqKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      X.size(), size, X.data<float>(), M->data<float>(), S->mutable_data<float>());
    SqrtKernel<<<CAFFE_GET_BLOCKS(S->size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      S->size(), (float)size, S->mutable_data<float>());
  }

  return true;
}

namespace {

REGISTER_CUDA_OPERATOR(MeanStdev, MeanStdevOp<float, CUDAContext>);

}  // namespace

}  // namespace caffe2
