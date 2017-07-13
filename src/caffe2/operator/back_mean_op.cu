#include "caffe2/operator/back_mean_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

int back_mean_strip(std::vector<TIndex> &dims, int count) {
  auto size = 1;
  while (count--) {
    size *= dims.back();
    dims.pop_back();
  }
  return size;
}

namespace {

__global__ void ZeroKernel(const int N, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
   Y[i] = 0;
  }
}

__global__ void BackMeanKernel(const int N, const int C, const float D, const float* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i / C] += X[i] / D;
  }
}

}  // namespace

template <>
bool BackMeanOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto dims = X.dims();
  auto size = back_mean_strip(dims, count_);
  Y->Resize(dims);
  if (Y->size() > 0) {
    ZeroKernel<<<CAFFE_GET_BLOCKS(Y->size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      Y->size(), Y->mutable_data<float>());
    BackMeanKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      X.size(), size, (float)size, X.data<float>(), Y->mutable_data<float>());
  }
  return true;
}

namespace {

__global__ void BackMeanGradientKernel(const int N, const int C, const float D, const float* dY, float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i / C] / D;
  }
}

}  // namespace

template <>
bool BackMeanGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  auto dims = X.dims();
  auto size = back_mean_strip(dims, count_);
  DCHECK_EQ(dY.size() * size, dX->size());
  if (dY.size() > 0) {
    BackMeanGradientKernel<<<CAFFE_GET_BLOCKS(dX->size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      dX->size(), size, (float)size, dY.data<float>(), dX->mutable_data<float>());
  }
  return true;
}


namespace {

REGISTER_CUDA_OPERATOR(BackMean, BackMeanOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(BackMeanGradient, BackMeanGradientOp<float, CUDAContext>);

}  // namespace

}  // namespace caffe2
