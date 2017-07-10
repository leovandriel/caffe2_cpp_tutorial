#ifndef MEAN_STDEV_OP_H
#define MEAN_STDEV_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class MeanStdevOp final : public Operator<Context> {
 public:
  MeanStdevOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // MEAN_STDEV_OP_H
