#ifndef OPERATOR_COUT_H
#define OPERATOR_COUT_H

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
    }
    std::cout << std::endl;
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

#endif  // OPERATOR_COUT_H
