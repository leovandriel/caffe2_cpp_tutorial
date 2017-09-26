#ifndef OPERATOR_ZERO_ONE_OP_H
#define OPERATOR_ZERO_ONE_OP_H

#include <caffe2/core/operator.h>

namespace caffe2 {

template <typename T, class Context>
class ZeroOneOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ZeroOneOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(PREDICTION, LABEL);
};

}  // namespace caffe2

#endif  // OPERATOR_ZERO_ONE_OP_H
