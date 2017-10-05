#ifndef OPERATOR_SQUARED_L2_OP_H
#define OPERATOR_SQUARED_L2_OP_H

#include <caffe2/core/operator.h>

namespace caffe2 {

template <typename T, class Context>
class SquaredL2Op final : public Operator<Context> {
 public:
  SquaredL2Op(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int count_;
};

template <typename T, class Context>
class SquaredL2GradientOp final : public Operator<Context> {
 public:
  SquaredL2GradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int count_;
};

}  // namespace caffe2

#endif  // OPERATOR_SQUARED_L2_OP_H
