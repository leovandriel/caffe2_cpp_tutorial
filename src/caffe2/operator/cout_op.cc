#include "caffe2/operator/cout_op.h"

#include "caffe2/util/tensor.h"

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 {

void print(const TensorCPU &tensor, const std::string &name, int max = 100) {
  const auto &data = tensor.data<float>();
  if (name.length() > 0) std::cout << name << "(" << tensor.dims() << "): ";
  for (auto i = 0; i < (tensor.size() > max ? max : tensor.size()); ++i) {
    std::cout << (float)data[i] << ' ';
  }
  if (tensor.size() > max) {
    std::cout << "... (" << *std::min_element(data, data + tensor.size()) << ","
              << *std::max_element(data, data + tensor.size()) << ")";
  }
  if (name.length() > 0) std::cout << std::endl;
}

template <>
bool CoutOp<CPUContext>::RunOnDevice() {
  auto index = 0;
  for (auto &title : titles) {
    print(Input(index++), title);
  }
  return true;
}

#ifdef WITH_CUDA
template <>
bool CoutOp<CUDAContext>::RunOnDevice() {
  auto index = 0;
  for (auto &title : titles) {
    print(TensorCPU(Input(index++)), title);
  }
  return true;
}
#endif

REGISTER_CPU_OPERATOR(Cout, CoutOp<CPUContext>);

#ifdef WITH_CUDA
REGISTER_CUDA_OPERATOR(Cout, CoutOp<CUDAContext>);
#endif

OPERATOR_SCHEMA(Cout)
    .NumInputs(1, 10)
    .NumOutputs(0)
    .SetDoc("Log tensor to std::cout")
    .Input(0, "tensor", "The tensor to log");

SHOULD_NOT_DO_GRADIENT(Cout);

}  // namespace caffe2
