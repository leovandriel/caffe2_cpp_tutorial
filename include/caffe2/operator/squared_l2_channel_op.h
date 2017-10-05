#ifndef OPERATOR_SQUARED_L2_CHANNEL_OP_H
#define OPERATOR_SQUARED_L2_CHANNEL_OP_H

#include <caffe2/core/operator.h>

namespace caffe2 {

template <typename T, class Context>
class SquaredL2ChannelOp final : public Operator<Context> {
 public:
  SquaredL2ChannelOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        channel_(OperatorBase::GetSingleArgument<int>("channel", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int channel_;
};

template <typename T, class Context>
class SquaredL2ChannelGradientOp final : public Operator<Context> {
 public:
  SquaredL2ChannelGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        channel_(OperatorBase::GetSingleArgument<int>("channel", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int channel_;
};

}  // namespace caffe2

#endif  // OPERATOR_SQUARED_L2_CHANNEL_OP_H
