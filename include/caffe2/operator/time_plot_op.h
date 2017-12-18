#ifndef OPERATOR_TIME_PLOT_OP
#define OPERATOR_TIME_PLOT_OP

#include <caffe2/core/operator.h>

namespace caffe2 {

template <typename T, class Context>
class TimePlotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TimePlotOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        window_(OperatorBase::GetSingleArgument<std::string>("window",
                                                             def.input(0))),
        label_(OperatorBase::GetSingleArgument<std::string>("label",
                                                            def.input(0))),
        step_(OperatorBase::GetSingleArgument<int>("step", 1)),
        index_(0),
        step_count_(0),
        value_sum_(0.f),
        sq_sum_(0.f),
        index_sum_(0.f) {}
  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(DATA, ITER);
  std::string label_;
  std::string window_;
  int step_;
  int index_;

  int step_count_;
  float value_sum_;
  float sq_sum_;
  float index_sum_;
};

}  // namespace caffe2

#endif  // OPERATOR_TIME_PLOT_OP
