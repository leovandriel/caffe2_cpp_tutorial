#ifndef OPERATOR_TIME_PLOT_OP
#define OPERATOR_TIME_PLOT_OP

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class TimePlotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TimePlotOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        name_(OperatorBase::GetSingleArgument<std::string>("name", "default")),
        labels_(def.input_size()),
        pasts_(def.input_size()) {
    for (auto i = 0; i < labels_.size(); i++) {
      labels_[i] = def.input(i);
    }
  }
  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(VALUES);
  std::string name_;
  std::vector<std::string> labels_;
  std::vector<std::vector<float>> pasts_;
};

}  // namespace caffe2

#endif  // OPERATOR_TIME_PLOT_OP
