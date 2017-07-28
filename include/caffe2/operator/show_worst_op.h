#ifndef OPERATOR_SHOW_WORST_OP
#define OPERATOR_SHOW_WORST_OP

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ShowWorstOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ShowWorstOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(PREDICTION, LABEL, DATA, ITER);
};

}  // namespace caffe2

#endif  // OPERATOR_SHOW_WORST_OP
