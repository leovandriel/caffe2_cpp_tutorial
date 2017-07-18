#include "caffe2/operator/mean_stdev_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

__global__ void MeanStdevKernel(const int N, const int C, const float D, const float* X, float* M, float* S) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float sum = 0;
    for (int j = i * C, e = j + C; j != e; j++) {
      sum += X[j];
    }
    M[i] = sum / D;
    float sumsq = 0;
    for (int j = i * C, e = j + C; j != e; j++) {
      float v = X[j] - M[i];
      sumsq += v * v;
    }
    S[i] = sqrtf(sumsq / D);
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
    MeanStdevKernel<<<CAFFE_GET_BLOCKS(M->size()), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      M->size(), size, (float)size, X.data<float>(), M->mutable_data<float>(), S->mutable_data<float>());
  }

  return true;
}

namespace {

REGISTER_CUDA_OPERATOR(MeanStdev, MeanStdevOp<float, CUDAContext>);

}  // namespace

}  // namespace caffe2
