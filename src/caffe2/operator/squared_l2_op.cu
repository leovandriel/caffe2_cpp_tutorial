#include <caffe2/core/context_gpu.h>
#include "caffe2/operator/squared_l2_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SquaredL2Kernel(const int N, const int D, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float sum = 0;
    for (int j = i * D, e = j + D; j != e; j++) {
      float x = X[j];
      sum += x * x;
    }
    Y[i] = sum;
  }
}

}  // namespace

template <>
bool SquaredL2Op<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.size() / N : 0;
  Y->Resize(vector<TIndex>(size_t(1), N));
  SquaredL2Kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
                    context_.cuda_stream()>>>(N, D, X.data<float>(),
                                              Y->mutable_data<float>());
  return true;
}

namespace {
template <typename T>
__global__ void SquaredL2KernelKernel(const int N, const int D, const T* X,
                                      const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N * D) { dX[i] = X[i] * dY[i / D]; }
}
}  // namespace

template <>
bool SquaredL2GradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.size() / N : 0;
  CAFFE_ENFORCE_EQ(dY.ndim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), N);
  dX->ResizeLike(X);
  SquaredL2KernelKernel<float>
      <<<CAFFE_GET_BLOCKS(N * D), CAFFE_CUDA_NUM_THREADS, 0,
         context_.cuda_stream()>>>(N, D, X.data<float>(), dY.data<float>(),
                                   dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(SquaredL2, SquaredL2Op<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SquaredL2Gradient,
                       SquaredL2GradientOp<float, CUDAContext>);

}  // namespace caffe2
