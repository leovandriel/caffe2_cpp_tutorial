#ifndef OPERATOR_SHOW_WORST_OP
#define OPERATOR_SHOW_WORST_OP

#include <caffe2/core/operator.h>

namespace caffe2 {

template <typename T, class Context>
class ShowWorstOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ShowWorstOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.0)),
        mean_(OperatorBase::GetSingleArgument<float>("mean", 128.0)),
        under_name_(OperatorBase::GetSingleArgument<std::string>(
            "under_name", "undercertain")),
        over_name_(OperatorBase::GetSingleArgument<std::string>(
            "over_name", "overcertain")) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(PREDICTION, LABEL, DATA, ITER);
  float scale_;
  float mean_;
  const std::string under_name_;
  const std::string over_name_;
};

}  // namespace caffe2

#endif  // OPERATOR_SHOW_WORST_OP
