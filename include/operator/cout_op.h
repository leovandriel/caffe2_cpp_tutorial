#ifndef COUT_OP_H
#define COUT_OP_H

#include "caffe2/core/operator.h"

#include "util/print.h"

namespace caffe2 {

template <class Context>
class CoutOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CoutOp(const OperatorDef& def, Workspace* ws)
    : Operator<Context>(def, ws),
      titles(def.input_size()) {
    for (auto i = 0; i < titles.size(); i++) {
        titles[i] = def.input(i);
    }
  }

  bool RunOnDevice() override {
    auto index = 0;
    for (auto &title: titles) {
      auto tensor = Input(index++);
      std::cout << title << " (" << tensor.dims() << "): ";
      print(tensor);
      std::cout << std::endl;
    }
    return true;
  }

 protected:
  std::vector<string> titles;
};

namespace {

REGISTER_CPU_OPERATOR(Cout, CoutOp<CPUContext>);

OPERATOR_SCHEMA(Cout)
  .NumInputs(1, 10)
  .NumOutputs(0)
  .SetDoc("Log tensor to std::cout")
  .Input(0, "tensor", "The tensor to log");

SHOULD_NOT_DO_GRADIENT(Cout);

} // namespace

}  // namespace caffe2

#ifdef WITH_CUDA

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"

namespace caffe2 {

template <typename T>
class CuDNNCoutOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNCoutOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
      titles(def.input_size()) {
    for (auto i = 0; i < titles.size(); i++) {
        titles[i] = def.input(i);
    }
  }

  bool RunOnDevice() override {
    auto index = 0;
    for (auto &title: titles) {
      auto tensor = TensorCPU(Input(index++));
      std::cout << title << " (" << tensor.dims() << "): ";
      print(tensor);
      std::cout << std::endl;
    }
    return true;
  }

 protected:
  std::vector<string> titles;
};

namespace {

REGISTER_CUDNN_OPERATOR(Cout, CuDNNCoutOp<float>);

}  // namespace

}  // namespace caffe2

#endif

#endif  // COUT_OP_H
