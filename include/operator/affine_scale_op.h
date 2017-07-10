#ifndef AFFINE_SCALE_OP_H
#define AFFINE_SCALE_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class AffineScaleOp final : public Operator<Context> {
 public:
  AffineScaleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      inverse_(OperatorBase::GetSingleArgument<int>("inverse", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int inverse_;
};

template <typename T, class Context>
class AffineScaleGradientOp final : public Operator<Context> {
 public:
  AffineScaleGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      inverse_(OperatorBase::GetSingleArgument<int>("inverse", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int inverse_;
};

} // namespace caffe2

#endif // AFFINE_SCALE_OP_H
